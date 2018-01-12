import datetime
import logging
import sqlite3

import trading.config
import trading.utils

class DbValue:
    """Db used to store data."""

    DATE_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'

    def __init__(self, db_name):
        """Class Initialisation."""

        self.db_file = '{}.sqlite'.format(trading.config.conf.database_file)
        self.name = db_name

        self.cnx = sqlite3.connect(self.db_file, check_same_thread=False)
        self.cursor = self.cnx.cursor()

        self.current_transaction = False

        self.cursor.execute('CREATE table IF NOT EXISTS {}'
                            '(dateTime DATETIME, value REAL)'.format(self.name))
        self.query = None
        self.cnx.commit()

    @trading.utils.database_query
    def insert_value(self, date_time, value):
        """Insert value into db.

        :param datetime.datetime date_time: date_time
        :param value: value
        """

        logging.debug('')

        time_print = date_time.strftime(self.DATE_TIME_FORMAT)
        self.query = ('INSERT INTO {} (dateTime, value) '
                      'VALUES (?,?)'.format(self.name))

        self.cursor.execute(self.query, [time_print, value])
        
        self.cnx.commit()
        logging.info('%s date_time: %s value: %s', self.name, time_print, value)
        return True

    @trading.utils.database_query
    def get_values(self, start_date_time, stop_date_time):
        """Get values."""

        logging.debug('')
        self.query = ('SELECT dateTime, value FROM {} '
                      'where dateTime BETWEEN ? AND ?'.format(self.name))
        self.cursor.execute(self.query,
                            [start_date_time.strftime(self.DATE_TIME_FORMAT),
                             stop_date_time.strftime(self.DATE_TIME_FORMAT)])

        (values) = self.cursor.fetchall()

        if values is None or len(values) == 0:
            raise NoValues

        if values is not None and len(values) != 0:

            logging.debug('%s values between [%s,%s]: %s',
                          self.name,
                          start_date_time.strftime(self.DATE_TIME_FORMAT),
                          stop_date_time.strftime(self.DATE_TIME_FORMAT),
                          ','.join('date_time {} value: {}'.format(date_time,
                                                                   value)
                                   for date_time, value in values)
                          )

            return [value[1] for value in values]
        return None

    @trading.utils.database_query
    def get_last_values(self, nb_values=-1):
        """Get last values."""

        logging.debug('nb_values: %d', nb_values)
        self.query = ('SELECT dateTime, value FROM {} '
                      'ORDER BY dateTime DESC'.format(self.name))

        if nb_values >= 1:
            self.query = self.query + ' LIMIT ?'
            self.cursor.execute(self.query, [nb_values])
        elif nb_values == -1:
            self.cursor.execute(self.query)

        (values) = self.cursor.fetchall()

        logging.debug('Last %d values of %s: %s',
                     nb_values,
                     self.name,
                     ','.join('dateTime: {}, value: {}'.format(date_time, value)
                              for date_time, value in values)
                     )
        if values is not None and len(values) != 0:
            values = [value[1] for value in values]
            values.reverse()
            return values
        return None

    def reset(self):
        """Reset database."""
        logging.info('%s', self.name)

        self.cursor.execute('DELETE FROM ' + self.name)
        self.cnx.commit()
        self.current_transaction = False
