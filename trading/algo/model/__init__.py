#! /usr/bin/env python

#import trading.algo.model.rolling_mean_pricing as rmp
#import trading.algo.model.pricing as pricing
from .pricing import Pricing
from .rolling_mean_pricing import RollingMeanPricing


def create():
    Pricing.createTable(ifNotExists=True)
    RollingMeanPricing.createTable(ifNotExists=True)