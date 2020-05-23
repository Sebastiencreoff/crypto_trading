#!/usr/bin/env python

import datetime
import json
import logging

from . import model
from . import average
from . import bollinger


class AlgoMain:
    """Class which manage all algorithm to deal with data."""

    def __init__(self, config_dict):
        """Class Initialisation."""

        self.__dict__ = json.load(open(config_dict, mode='r'))
        self.algo_ifs = []
        self.algo_ifs.append(average.GuppyMMA(config_dict))
        self.algo_ifs.append(bollinger.Bollinger(config_dict))

        self.max_frequencies = max(x.max_frequencies()
                                   for x in self.algo_ifs
                                   if x.max_frequencies())
        model.create()

    def process(self, current_value, currency):
        """Process data, it returned 1 to buy and -1 to sell."""

        # Price data
        model.pricing.Pricing(currency=currency,
                              date_time=datetime.datetime.now(),
                              value=current_value)

        values = model.pricing.get_last_values(
            count=self.max_frequencies,
            currency=currency)

        result = 0
        for algo in self.algo_ifs:
            result += algo.process(current_value, values, currency)

        logging.info('result: %d', result)

        return result

    def reset(self):
        model.reset()

