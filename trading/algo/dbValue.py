import sqlite3

class DbValue:
    """ Db used to store data """

    def __init__(self, dbFile):
        """ Initialisation of class """

        self.cnx = sqlite3.connect(dbFile)
        self.dbFile = dbFile

    def insertValue(self, value, dateTime = 'NOW'):
        """ Insert value into db """
        pass

    def getValues(self, startDateTime, stopDateTime, frequency):
        """ Get values """
        pass