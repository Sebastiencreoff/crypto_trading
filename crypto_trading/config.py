#!/usr/bin/env python
# coding: utf-8

import json
import logging
import os

import sqlobject


class Config:
    def __init__(self, config_file):
        self.config_dict = json.load(open(
            os.path.abspath(config_file), mode='r'))

        try:
            self.dir_path = os.path.dirname(config_file) + '/'
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

    def setup_db(self):
        sqlobject.sqlhub.processConnection = \
            sqlobject.dbconnection.connectionForURI(
                'sqlite:{}'.format(os.path.abspath(self.database_file)))
        self.db_conn = sqlobject.sqlhub.processConnection


def init(config_file):

    conf = Config(config_file)
    conf.setup_db()
    return conf




