#! /usr/bin/env python

from .pricing import Pricing
from .rolling_mean_pricing import RollingMeanPricing
from .guppy_mma import Guppy


def create():
    Pricing.createTable(ifNotExists=True)
    RollingMeanPricing.createTable(ifNotExists=True)
    Guppy.createTable(ifNotExists=True)
