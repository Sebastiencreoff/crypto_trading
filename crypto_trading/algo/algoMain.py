#!/usr/bin/env python

import datetime
import json
import logging

from . import model
from . import average
import crypto_trading.config as cfg


class AlgoMain:
    """Class which manage all algorithm to deal with data."""

    def __init__(self, config_dict):
        """Class Initialisation."""

        self.__dict__ = json.load(open(config_dict, mode='r'))
        self.algo_ifs = []
        self.algo_ifs.append(average.GuppyMMA(config_dict))

        model.create()

    def process(self, currency_value, currency):
        """Process data, it returned 1 to buy and -1 to sell."""

        # Price data
        model.pricing.Pricing(currency=currency,
                              date_time=datetime.datetime.now(),
                              value=currency_value)
        result = 0
        for algo in self.algo_ifs:
            result += algo.process(currency_value, currency)

        logging.info('result: %d', result)

        return result


