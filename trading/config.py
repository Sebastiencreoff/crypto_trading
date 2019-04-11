#!/usr/bin/env python
# coding: utf-8

import json
import logging
import os
import sys

import sqlobject


conf = None


class Config:
    def __init__(self, config_file):
        self.config_dict = json.load(open(config_file, mode='r'))

        try:
            self.database_file = self.config_dict['database_file']
            self.currency = self.config_dict['currency']
            self.transaction_amt = self.config_dict['transactionAmt']
            self.connection = self.config_dict['connection']
            self.connection_config = self.config_dict['connectionConfig']
            self.algo_config = self.config_dict['algoConfig']
            self.delay = self.config_dict['delay']

            # Database  name
            self.pricing = 'Pricing'

        except KeyError:
            logging.exception('error in configuration file')

    def setup_db(self):
        sqlobject.sqlhub.processConnection = \
            sqlobject.dbconnection.connectionForURI(
                'sqlite:{}'.format(os.path.abspath(self.database_file)))


def init(config_file):

    global conf
    conf = Config(config_file)
    conf.setup_db()




