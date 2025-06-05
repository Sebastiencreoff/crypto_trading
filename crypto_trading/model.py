#!/usr/bin/env python

import datetime
import logging

import sqlobject

# New SQLObject class for Price History
class PriceTick(sqlobject.SQLObject):
    currency = sqlobject.col.StringCol()
    timestamp = sqlobject.col.DateTimeCol()
    price = sqlobject.col.FloatCol()

    # Optional: Index for faster queries
    # class sqlmeta:
    #     indices = [
    #         sqlobject.dbIndex('currency', 'timestamp'),
    #     ]

def create(db_conn): # Modified signature
    """Creates tables if they don't exist, using the provided db_conn."""
    Trading.createTable(ifNotExists=True, connection=db_conn)
    PriceTick.createTable(ifNotExists=True, connection=db_conn) # Added PriceTick table creation
    logging.info("Database tables (Trading, PriceTick) ensured to exist.")


def reset(db_conn, currency=None): # Modified signature
    """Reset database for a specific currency or all if currency is None."""
    if currency:
        Trading.deleteMany(Trading.q.currency == currency, connection=db_conn)
        PriceTick.deleteMany(PriceTick.q.currency == currency, connection=db_conn) # Also reset PriceTick
        logging.info(f"Database reset for currency: {currency}")
    else:
        # This path might be dangerous / unintended if called without currency now.
        # Consider if full reset is ever needed or should be prevented/logged heavily.
        Trading.deleteMany(True, connection=db_conn) # Deletes all Trading records
        PriceTick.deleteMany(True, connection=db_conn) # Deletes all PriceTick records
        logging.warning("Full database reset performed for Trading and PriceTick tables.")


def get_current_trading(db_conn, currency=None): # Modified signature
    """Get current transaction (already bought but not yet sold) for a given currency."""
    if not currency:
        # This case should ideally not be hit if called from Trading class logic,
        # as currency should always be specified.
        logging.error("get_current_trading called without specifying currency. This is not recommended.")
        # Fallback to original behavior if absolutely necessary, but this might fetch unrelated trades.
        # trade = Trading.select(Trading.q.sell_date_time == None, connection=db_conn)
        return None # Prefer returning None if currency is missing.

    trade = Trading.select(
        sqlobject.AND(Trading.q.sell_date_time == None, Trading.q.currency == currency),
        connection=db_conn
    )

    if trade.count():
        logging.info(f"Current open transaction found for currency {currency}: Bought at {trade[0].buy_date_time}, ID: {trade[0].id}")
        return trade[0]
    logging.info(f"No current open transaction found for currency {currency}.")
    return None

def get_profits(db_conn, currency=None): # Modified signature
    """Get total profits for a given currency, or all currencies if None."""
    if currency:
        selected_trades = Trading.select(
            sqlobject.AND(Trading.q.profit != None, Trading.q.currency == currency),
            connection=db_conn
        )
        logging.info(f"Calculating profits for currency: {currency}")
    else:
        # This case might be used for global profit calculation if ever needed.
        selected_trades = Trading.select(Trading.q.profit != None, connection=db_conn)
        logging.info("Calculating profits for all currencies.")

    total_profit = sum(trade.profit for trade in selected_trades)
    logging.info(f"Total calculated profit: {total_profit}")
    return total_profit

# New function to save price ticks
def save_price(db_conn, currency_pair, price_value, ts):
    """Saves a price tick to the database using the provided db_conn."""
    try:
        # Ensure timestamp is a datetime object
        timestamp_dt = datetime.datetime.fromtimestamp(ts) if not isinstance(ts, datetime.datetime) else ts

        PriceTick(currency=currency_pair, price=price_value, timestamp=timestamp_dt, connection=db_conn)
        # SQLObject creates the record upon instantiation when a connection is provided.
        logging.debug(f"Saved price for {currency_pair}: {price_value} at {timestamp_dt}")
    except Exception as e:
        logging.error(f"Error saving price for {currency_pair} ({price_value} at {ts}): {e}", exc_info=True)


class Trading(sqlobject.SQLObject):
    # Definition remains the same. Connection is handled at instance level.
    buy_date_time = sqlobject.col.DateTimeCol(default=None)
    buy_value = sqlobject.col.FloatCol() # This is the total EUR amount for the buy order
    buy_value_fee = sqlobject.col.FloatCol(default=None)
    currency_buy_amt = sqlobject.col.FloatCol(default=None) # Amount of crypto bought
    currency_buy_value = sqlobject.col.FloatCol(default=None) # Price per unit of crypto at buy
    currency_sell_value = sqlobject.col.FloatCol(default=None) # Price per unit of crypto at sell
    currency = sqlobject.col.StringCol() # e.g., "BTC/EUR"
    profit = sqlobject.col.FloatCol(default=None)
    sell_value_fee = sqlobject.col.FloatCol(default=None)
    sell_date_time = sqlobject.col.DateTimeCol(default=None)

    # Optional: Add index for currency and sell_date_time for faster queries in get_current_trading
    # class sqlmeta:
    #     indices = [
    #         sqlobject.dbIndex('currency', 'sell_date_time'),
    #     ]

    def save_buy(self, currency_amt, ref_fee_amt, buy_timestamp=None): # Added optional buy_timestamp
        """Saves buy transaction details.
           Connection is already associated with the instance.
        """
        if not currency_amt or ref_fee_amt is None: # ref_fee_amt can be 0
            logging.error(f"Error on buying {self.buy_value} of {self.currency} at {self.buy_date_time}: Invalid currency_amt or ref_fee_amt.")
            # Should this raise an error or return None?
            # Returning None might mask issues. Consider raising ValueError.
            return None

        self.buy_date_time = buy_timestamp if buy_timestamp else datetime.datetime.now()
        self.buy_value_fee = ref_fee_amt
        self.currency_buy_amt = currency_amt
        if self.currency_buy_amt == 0: # Avoid division by zero
            logging.error(f"Cannot calculate currency_buy_value: currency_buy_amt is zero for {self.currency} trade.")
            return None
        self.currency_buy_value = self.buy_value / self.currency_buy_amt # Price per crypto unit

        logging.info(f"BUY: {self.currency_buy_amt} {self.currency} at {self.currency_buy_value:.2f} (Total: {self.buy_value:.2f}, Fee: {self.buy_value_fee:.2f}) on {self.buy_date_time}. ID: {self.id if hasattr(self, 'id') else 'New'}")
        return self

    def save_sell(self, currency_sell_price, sell_fee_amt, sell_timestamp=None): # Added optional sell_timestamp
        """Saves sell transaction details and calculates profit.
           Connection is already associated with the instance.
        """
        if currency_sell_price is None or sell_fee_amt is None: # Can be 0
            logging.error(f"Error on selling {self.currency_buy_amt} {self.currency} (bought at {self.currency_buy_value}): Invalid currency_sell_price or sell_fee_amt.")
            return None

        if not self.currency_buy_amt or not self.buy_value:
            logging.error(f"Cannot save sell for {self.currency} trade {self.id if hasattr(self, 'id') else 'Unknown'}: Missing critical buy information (amount or value).")
            return None

        total_sell_value_native_currency = currency_sell_price * self.currency_buy_amt

        self.profit = (total_sell_value_native_currency
                     - self.buy_value # Initial buy value in native currency (e.g. EUR)
                     - (self.buy_value_fee if self.buy_value_fee else 0)
                     - (sell_fee_amt if sell_fee_amt else 0))

        self.currency_sell_value = currency_sell_price # Price per crypto unit at sell
        self.sell_value_fee = sell_fee_amt
        self.sell_date_time = sell_timestamp if sell_timestamp else datetime.datetime.now()

        logging.info(f"SELL: {self.currency_buy_amt} {self.currency} at {self.currency_sell_value:.2f} (Total value: {total_sell_value_native_currency:.2f}, Fee: {self.sell_value_fee:.2f}) on {self.sell_date_time}. Profit: {self.profit:.2f}. ID: {self.id}")
        return self
