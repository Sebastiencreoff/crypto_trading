#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np


def is_increasing(values):
    function = np.polyfit(np.array([x for x in range(0, len(values))]),
                          np.array(values), 1)

    return function[0] > 0
