import sqlite3

class DbTransaction:
    """ Db used to store transaction data """

    def __init__(self, dbFile):
        """ Initialisation of class """

        self.cnx = sqlite3.connect(dbFile)
        self.dbFile = dbFile

        self.cursor = self.cnx.cursor()

        self.cursor.execute(
            "CREATE table IF NOT EXISTS %self.dbFile"
            " (currency CHAR(3),"
            " currencyValue REAL,"
            " buyValue REAL,"
            " sellValue REAL,"
            " buyDateTime DATETIME,"
            " sellDateTime DATETIME)"
            )

        self.cursor.commit()

    def buy(self, currency, value, buyValue, dateTime = 'NOW'):
        """ add transaction value into db """

        try:
            transaction = [ currency, value, buyValue, dateTime]

            self.cursor.execute("INSERT INTO %self.dbFile "
            " (currency, currencyValue, buyValue, buyDateTime) VALUES (?,?,?,?)", transaction)
        
        except sqlite3.Error as e:
            print ("An error occurred:", e.args[0])
            return False

        return True

    def sell(self, currency, buyDateTime, sellValue,  sellDateTime='NOW'):
        """ update transaction value into db """
        
        try:
            transaction = [ sellValue, sellDateTime, currency, buyDateTime]

            self.cursor.execute("UPDATE %self.dbFile "
            " SET sellValue = ?, sellDateTime = ? "
            " WHERE currency = ? AND buyDateTime = ? AND sellDateTime is null", transaction)
        
        except sqlite3.Error as e:
            print ("An error occurred:", e.args[0])
            return False

        return True


    def getCurrentTransaction(self, currency):
        """ Get current transaction """

        try:
            fetchColunm = (currency,)

            self.cursor.execute("SELECT buyDateTime FROM %self.dbFile "
            " WHERE currency = ? AND sellDateTime is null ", fetchColunm)
        
            return (True, self.cursor.fetchone())

        except sqlite3.Error as e:
            print ("An error occurred:", e.args[0])
            return False