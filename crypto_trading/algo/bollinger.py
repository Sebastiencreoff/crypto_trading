#! /usr/bin/env python
# -*- coding:utf-8 -*-

import json
import logging

from . import model


class Bollinger(object):

    __COUNT__ = 2
    __FREQUENCY__ = 20

    def __init__(self, config_dict):
        """Class Initialisation."""
        logging.debug('')

        cfg = json.load(open(config_dict, mode='r'))

        if cfg.get('Bollinger') is not None:
            self.frequency = cfg.get('Bollinger').get('frequency',
                                                      self.__FREQUENCY__)

        logging.info('Bollinger with a frequency at %s', self.frequency)

    def max_frequencies(self):
        return self.frequency

    def process(self, current_value, values, currency):
        """Process data, it returned 1 to buy and -1 to sell."""
        logging.debug('')

        model.bollinger.insert_value(currency, self.frequency, values)
        results = model.bollinger.get_last_values(
            currency, self.frequency,
            count=self.__COUNT__)

        if (len(values) > self.__COUNT__
                and all(x.lower_limit and x.upper_limit for x in results)):
            prev_bol, current_bol = results

            if (prev_bol.lower_limit > values[-2]
                    and current_bol.lower_limit <= values[-1]):
                logging.warning('Bollinger buy limit reached')
                return 1

            if (prev_bol.upper_limit < values[-2]
                    and current_bol.upper_limit >= values[-1]):
                logging.warning('Bollinger sell limit reached')
                return -1

        return 0