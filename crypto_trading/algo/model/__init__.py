#! /usr/bin/env python
# -*- coding:utf-8 -*-

from .pricing import Pricing
from .rolling_mean_pricing import RollingMeanPricing
from .guppy_mma import Guppy


def create():
    Pricing.createTable(ifNotExists=True)
    RollingMeanPricing.createTable(ifNotExists=True)
    Guppy.createTable(ifNotExists=True)
