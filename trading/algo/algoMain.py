#!/usr/bin/env python

import datetime
import json
import logging

import trading.algo.dbValue
import trading.algo.rollingMean


class AlgoMain:
    """Class which manage all algorithm to deal with data."""

    def __init__(self, config_dict):
        """Class Initialisation."""

        self.__dict__ = json.load(open(config_dict, mode='r'))
        self.db = trading.algo.dbValue.DbValue(trading.config.conf.pricing)

        self.algo_ifs = []
        self.algo_ifs.append(trading.algo.rollingMean.RollingMean(
            config_dict, self.db))

    def process(self, data_value):
        """Process data, it returned 1 to buy and -1 to sell."""

        # Price data

        self.db.insert_value(datetime.datetime.now(), data_value)

        result = 0
        for algo in self.algo_ifs:
            result = result + algo.process(data_value)

        if result:
            logging.warning('result: %d', result)

        return result


