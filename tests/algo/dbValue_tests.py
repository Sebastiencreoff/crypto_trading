#!/usr/bin/env python

import datetime

import nose.tools
import unittest

import trading.algo.dbValue


class DbValueTest(unittest.TestCase):
    
    def setup(self):
        print("SETUP!")
    
    def teardown(self):
        print("TEAR DOWN!")
    
    def test_getValues(self):

        db = trading.algo.dbValue.DbValue(db_file='dbValue', db_name='dbValue')

        db.reset()
        
        time = datetime.datetime.now()
        self.assertTrue(db.insert_value(time, 1))

        time = time + datetime.timedelta(minutes=1)
        self.assertTrue(db.insert_value(time, 2))

        time = time + datetime.timedelta(minutes=1)
        self.assertTrue(db.insert_value(time, 3))

        time = time + datetime.timedelta(minutes=1)
        self.assertTrue(db.insert_value(time, 4))

        # test getting value from out of datetime
        time_start = datetime.datetime.now() - datetime.timedelta(hours=1)
        time_end = time_start + datetime.timedelta(minutes=1)
        self.assertEqual((False, None), db.get_values(time_start, time_end))
        
        # test getting value from range dateTime and check the order
        time_start = datetime.datetime.now()
        time_end = time_start + datetime.timedelta(minutes=2)
        self.assertEqual((True, [1.0, 2.0, 3.0]),
                         db.get_values(time_start, time_end))
        
    def test_getLastValues(self):

        db = trading.algo.dbValue.DbValue(db_file="dbValue", db_name="dbValue")

        db.reset()
        
        # test getting value when database is empty

        time = datetime.datetime.now()
        self.assertTrue(db.insert_value(time, 1))

        time = time + datetime.timedelta(minutes=1)
        self.assertTrue(db.insert_value(time, 2))

        time = time + datetime.timedelta(minutes=1)
        self.assertTrue(db.insert_value(time, 3))

        time = time + datetime.timedelta(minutes=1)
        self.assertTrue(db.insert_value(time, 4))

        # test getting value with nbValue upper than database data
        self.assertEqual((True, [1.0, 2.0, 3.0, 4.0]), db.get_last_values(5))
        
        # test getting value with nbValue
        self.assertEqual((True, [2.0, 3.0, 4.0]), db.get_last_values(3))
