#! /usr/bin/env python
# -*- coding:utf-8 -*-

from .bollinger import Bollinger
from .guppy_mma import Guppy
from .pricing import Pricing
from .rolling_mean_pricing import RollingMeanPricing
from .security import MaxLost


def create():
    Bollinger.createTable(ifNotExists=True)
    Guppy.createTable(ifNotExists=True)
    MaxLost.createTable(ifNotExists=True)
    Pricing.createTable(ifNotExists=True)
    RollingMeanPricing.createTable(ifNotExists=True)

def reset():
    Bollinger.deleteMany(where=None)
    Guppy.deleteMany(where=None)
    MaxLost.deleteMany(where=None)
    Pricing.deleteMany(where=None)
    RollingMeanPricing.deleteMany(where=None)