#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

__version__ = '0.0.1'

from .slack_interface import SlackInterface
