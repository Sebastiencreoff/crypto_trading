#! /usr/bin/env python
# -*- coding:utf-8 -*-

from .pricing import Pricing
from .rolling_mean_pricing import RollingMeanPricing
from .guppy_mma import Guppy
from .security import Security


def create():
    Pricing.createTable(ifNotExists=True)
    RollingMeanPricing.createTable(ifNotExists=True)
    Guppy.createTable(ifNotExists=True)
    Security.createTable(ifNotExists=True)
