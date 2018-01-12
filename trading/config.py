#!/usr/bin/env python
# coding: utf-8

import json
import logging
import sys

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

        except KeyError as e:
            logging.exception('error in configuration file')


def init(config_file):

    global conf
    conf = Config(config_file)
