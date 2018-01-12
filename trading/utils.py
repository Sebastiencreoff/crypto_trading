#!/usr/bin/env python

import functools
import logging

import sqlite3


def database_query(function):
    """Generic decorator for database query."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except sqlite3.Error as e:
            logging.error('SQL Query Error "%s" (%s on %s)',
                           args[0].query, e.args[0], args[0].db_file)
            raise
    return wrapper
