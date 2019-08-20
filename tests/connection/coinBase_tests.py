#!/usr/bin/env python

import json
import tempfile

import unittest

import crypto_trading.connection.coinBase as coinbase


class CoinBaseConnectTest(unittest.TestCase):

    def coinBase_config(self):
        fp = tempfile.NamedTemporaryFile()
        fp.write(json.dumps({'api_key': 'xxxx',
                             'api_secret': 'xxx',
                             'simulation': True}).encode('utf-8'))
        fp.seek(0)
        return fp.name

    def test_unknownCurrency(self):

        connect = coinbase.CoinBaseConnect(self.coinBase_config())

        with self.assertRaises(NameError):
            connect.get_currency(ref_currency='Unknown Currency')

    def test_execute(self):

        connect = coinbase.CoinBaseConnect(self.coinBase_config())

        self.assertTrue(connect.get_currency())
        self.assertFalse(connect.in_progress())
        self.assertEqual(connect.buy_currency(amount=10), 0.1)
        self.assertTrue(connect.in_progress())
        self.assertEqual(connect.sell_currency(amount=10), True)

    def test_error(self):

        connect = coinbase.CoinBaseConnect(self.coinBase_config())

        # sell currency if no transaction processing
        self.assertFalse(connect.in_progress())
        self.assertEqual(connect.sell_currency(amount=10),  False)
       
        self.assertNotEqual(connect.buy_currency(amount=10), None)
        # buy currency if transaction processing
        self.assertTrue(connect.in_progress())
        self.assertEqual(connect.buy_currency(amount=10), None)

