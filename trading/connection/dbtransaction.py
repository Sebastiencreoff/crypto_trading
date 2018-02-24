import sqlite3
import datetime

class DbTransaction:
    """ Db used to store transaction data """

    def __init__(self, dbFile, currency):
        """ Initialisation of class """
        
        self.dbFile = dbFile + "." + currency
        self.cnx = sqlite3.connect(self.dbFile)
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

        self.currenTransaction = self.getCurrentTransaction()

    def buy(self,  value, buyValue):
        """ add transaction value into db """

        if self.currenTransaction == False:
            try:
                time = datetime.time.now().isoformat()
                transaction = [ value, buyValue, time]

                self.cursor.execute("INSERT INTO %self.dbFile "
                " ( currencyValue, buyValue, buyDateTime) VALUES (?,?,?)", transaction)
        
                self.cursor.commit()

                self.currenTransaction = True

                return (True, time)

            except sqlite3.Error as e:
                print ("An error occurred:", e.args[0])

        return False

    def sell(self,  buyDateTime, sellValue,  sellDateTime='NOW'):
        """ update transaction value into db """
        
        if self.currenTransaction == True:
            try:
                transaction = [ sellValue, sellDateTime, buyDateTime]

                self.cursor.execute("UPDATE %self.dbFile "
                " SET sellValue = ?, sellDateTime = ? "
                " WHERE  buyDateTime = ? AND sellDateTime is null", transaction)
        
                self.cursor.commit()

                self.currenTransaction == False
                return True

            except sqlite3.Error as e:
                print ("An error occurred:", e.args[0])

        return False

    def getCurrentTransaction(self):
        """ Get current transaction """

        try:

            self.cursor.execute("SELECT buyDateTime FROM %self.dbFile "
            " WHERE sellDateTime is null ", fetchColunm)
        
            return (True, self.cursor.fetchone())

        except sqlite3.Error as e:
            print ("An error occurred:", e.args[0])
            return False