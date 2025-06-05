#!/usr/bin/env python
# coding: utf-8

import json
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Removed: import sqlobject


class Config:
    def __init__(self, config_file_path): # Renamed config_file to config_file_path for clarity
        # Ensure config_file_path is absolute
        abs_config_file_path = os.path.abspath(config_file_path)

        with open(abs_config_file_path, mode='r') as f:
            self.config_data = json.load(f)

        try:
            # self.dir_path should be the directory of the config file itself
            self.dir_path = os.path.dirname(abs_config_file_path) + '/'

            # Ensure database_file path is absolute or relative to config file dir
            db_file_name = self.config_data['database_file']
            if not os.path.isabs(db_file_name):
                self.database_file = os.path.join(self.dir_path, db_file_name)
            else:
                self.database_file = db_file_name

            self.sqlalchemy_database_url = f"sqlite:///{os.path.abspath(self.database_file)}"

            self.currency = self.config_data['currency']
            self.transaction_amt = self.config_data['transactionAmt']
            self.connection_type = self.config_data['connection'] # Renamed from self.connection

            connection_cfg_name = self.config_data['connectionConfig']
            if not os.path.isabs(connection_cfg_name):
                self.connection_config_path = os.path.join(self.dir_path, connection_cfg_name)
            else:
                self.connection_config_path = connection_cfg_name

            # Load algo_config_dict from the file specified in main config
            algo_cfg_name = self.config_data['algoConfig']
            if not os.path.isabs(algo_cfg_name):
                algo_config_file_path = os.path.join(self.dir_path, algo_cfg_name)
            else:
                algo_config_file_path = algo_cfg_name

            if os.path.exists(algo_config_file_path):
                with open(algo_config_file_path, 'r') as f_algo:
                    self.algo_config_dict = json.load(f_algo)
            else:
                logging.warning(f"Algorithm configuration file not found: {algo_config_file_path}")
                self.algo_config_dict = {} # Default to empty dict if file not found

            self.delay_secs = self.config_data['delay'] # Renamed from self.delay

            # SQLAlchemy engine and session factory, initialized lazily
            self._engine = None
            self._session_factory = None

            # Removed: self.pricing = 'Pricing'
            # Removed: self.db_conn = None (SQLObject specific)

        except KeyError as e:
            logging.exception(f'Error in configuration file: Missing key {e}')
            raise # Re-raise the exception to make it clear config loading failed

    def get_engine(self):
        if self._engine is None:
            if not self.sqlalchemy_database_url:
                raise ValueError("SQLAlchemy database URL is not set.")
            logging.info(f"Creating SQLAlchemy engine for URL: {self.sqlalchemy_database_url}")
            self._engine = create_engine(self.sqlalchemy_database_url)
        return self._engine

    def get_session(self):
        if self._session_factory is None:
            engine = self.get_engine()
            self._session_factory = sessionmaker(bind=engine)
            logging.info("SQLAlchemy session factory created.")
        return self._session_factory()

    # Removed setup_db(self) method


def init(config_file_path): # Renamed config_file to config_file_path
    """Initializes and returns a Config object."""
    logging.info(f"Initializing configuration from: {config_file_path}")
    conf = Config(config_file_path)
    # Removed: conf.setup_db()
    # Optionally, can try to establish a DB connection here to catch errors early
    # try:
    #     engine = conf.get_engine()
    #     with engine.connect() as connection: # Test connection
    #         logging.info("Database connection successful via SQLAlchemy engine.")
    # except Exception as e:
    #     logging.error(f"Failed to connect to database using SQLAlchemy engine: {e}", exc_info=True)
    #     raise
    return conf
