#!/usr/bin/env python

import datetime
import json
import logging

import trading.algo.model as model
import trading.algo.average
import trading.config as cfg


class AlgoMain:
    """Class which manage all algorithm to deal with data."""

    def __init__(self, config_dict):
        """Class Initialisation."""

        self.__dict__ = json.load(open(config_dict, mode='r'))
        self.algo_ifs = []
        self.algo_ifs.append(trading.algo.average.GuppyMMA(config_dict))

        model.create()

    def process(self, currency_value):
        """Process data, it returned 1 to buy and -1 to sell."""

        # Price data
        model.pricing.Pricing(currency=cfg.conf.currency,
                              date_time=datetime.datetime.now(),
                              value=currency_value)
        result = 0
        for algo in self.algo_ifs:
            result += algo.process(currency_value)

        logging.info('result: %d', result)

        return result


