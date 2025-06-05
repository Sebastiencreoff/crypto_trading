#!/usr/bin/env python
# coding: utf-8

import json
import logging
import os
import boto3
from botocore.exceptions import ClientError
import sqlobject


class Config:
    def __init__(self, config_file):
        self.config_dict = json.load(open(
            os.path.abspath(config_file), mode='r'))

        try:
            self.use_rds = os.environ.get('USE_RDS', 'False').lower() == 'true'
            self.db_secret_name = os.environ.get('DB_SECRET_NAME')
            self.dir_path = os.path.dirname(config_file) + '/'
            if not self.use_rds:
                self.database_file = self.config_dict['database_file']
            self.currency = self.config_dict['currency']
            self.transaction_amt = self.config_dict['transactionAmt']
            self.connection = self.config_dict['connection']
            self.connection_config = self.dir_path + self.config_dict['connectionConfig']
            self.algo_config = self.dir_path + self.config_dict['algoConfig']
            self.delay = self.config_dict['delay']

            # Database  name
            self.pricing = 'Pricing'
            self.db_conn = None
        except KeyError:
            logging.exception('error in configuration file')

    def get_db_credentials_from_secrets_manager(self):
        """Fetch DB credentials from AWS Secrets Manager."""
        if not self.db_secret_name:
            logging.error("DB_SECRET_NAME environment variable is not set.")
            raise ValueError("DB_SECRET_NAME not set for RDS configuration.")

        logging.info(f"Fetching DB credentials from Secrets Manager: {self.db_secret_name}")
        client = boto3.client('secretsmanager')
        try:
            get_secret_value_response = client.get_secret_value(SecretId=self.db_secret_name)
        except ClientError as e:
            logging.error(f"Error fetching secret: {e}")
            raise
        else:
            if 'SecretString' in get_secret_value_response:
                secret = get_secret_value_response['SecretString']
                return json.loads(secret)
            else:
                # Handle binary secret if needed, though unlikely for DB creds
                logging.error("SecretString not found in AWS Secrets Manager response.")
                raise ValueError("SecretString not found in AWS Secrets Manager response.")

    def setup_db(self):
        if self.use_rds:
            logging.info("Using RDS configuration.")
            creds = self.get_db_credentials_from_secrets_manager()
            db_uri = f"postgresql://{creds['username']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['dbname']}"
            logging.info(f"RDS DB URI: postgresql://{creds['username']}:****@{creds['host']}:{creds['port']}/{creds['dbname']}")
        else:
            logging.info("Using SQLite configuration.")
            if not hasattr(self, 'database_file') or not self.database_file:
                logging.error("Database file not configured for SQLite.")
                raise ValueError("Database file not configured for SQLite.")
            db_uri = 'sqlite:{}'.format(os.path.abspath(self.database_file))
            logging.info(f"SQLite DB URI: {db_uri}")

        sqlobject.sqlhub.processConnection = sqlobject.dbconnection.connectionForURI(db_uri)
        self.db_conn = sqlobject.sqlhub.processConnection


def init(config_file):

    conf = Config(config_file)
    conf.setup_db()
    return conf




