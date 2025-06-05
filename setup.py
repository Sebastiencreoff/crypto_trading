#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import crypto_trading

setup(name='crypto_trading',
      version=crypto_trading.__version__,
      description='Crypto Trading project with CoinBase API',
      url='https://github.com/Sebastiencreoff/pythonTrading',
      author='Sebastien Creoff',
      author_email='sebastien.creoff@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      entry_points={
        'console_scripts': [
            'trading = crypto_trading.main:main',
        ],
      },
      install_requires=['setuptools',
                        'coinbase',
                        'numpy',
                        'pandas',
                        'requests',
                        'sqlobject',
                        'psycopg2-binary',
                        'boto3',
                        'python-binance>=1.0.16'
      ],
      tests_require=[
          'nose'
      ],
      )
