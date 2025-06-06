import datetime
import logging
from sqlalchemy import func
from sqlalchemy.orm import Session

# Assuming models are in .models relative to this file if this file is in crypto_trading/database/
# If this file is crypto_trading/core_operations.py, then from .database.models
# Based on current plan, models are in crypto_trading/database/models.py
# and this file will be crypto_trading/database/core_operations.py
from .models import TradingTransaction, PriceTick

logger = logging.getLogger(__name__)

def save_price_tick(session: Session, currency_pair: str, price: float, timestamp: datetime.datetime = None):
    """Saves a new price tick to the database."""
    if timestamp is None:
        timestamp = datetime.datetime.utcnow()

    try:
        price_tick_entry = PriceTick(
            currency_pair=currency_pair,
            price=price,
            timestamp=timestamp
        )
        session.add(price_tick_entry)
        session.commit() # Or commit can be handled by the caller context
        logger.debug(f"Saved price tick for {currency_pair}: {price} at {timestamp}")
        return price_tick_entry
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving price tick for {currency_pair}: {e}", exc_info=True)
        raise

def get_open_transaction(session: Session, currency_pair: str, task_id: str = None) -> TradingTransaction | None:
    """
    Retrieves the current open (bought but not yet sold) transaction
    for a given currency pair and optionally a specific task_id.
    Returns the TradingTransaction object or None.
    """
    try:
        query = session.query(TradingTransaction).filter(
            TradingTransaction.currency_pair == currency_pair,
            TradingTransaction.sell_date_time == None
        )
        if task_id:
            query = query.filter(TradingTransaction.task_id == task_id)

        open_transaction = query.order_by(TradingTransaction.buy_date_time.desc()).first()

        if open_transaction:
            logger.info(f"Found open transaction for {currency_pair} (Task ID: {task_id if task_id else 'any'}): {open_transaction.id}")
        return open_transaction
    except Exception as e:
        logger.error(f"Error getting open transaction for {currency_pair}: {e}", exc_info=True)
        raise


def get_total_profit(session: Session, currency_pair: str = None, task_id: str = None) -> float:
    """
    Calculates the total profit from all completed transactions.
    Can be filtered by currency_pair and/or task_id.
    """
    try:
        query = session.query(func.sum(TradingTransaction.profit_eur)).filter(
            TradingTransaction.profit_eur != None
        )
        if currency_pair:
            query = query.filter(TradingTransaction.currency_pair == currency_pair)
        if task_id:
            query = query.filter(TradingTransaction.task_id == task_id)

        total_profit = query.scalar()
        return total_profit if total_profit is not None else 0.0
    except Exception as e:
        logger.error(f"Error calculating total profit: {e}", exc_info=True)
        raise

def reset_trading_transactions(session: Session, currency_pair: str = None, task_id: str = None):
    """
    Deletes trading transactions. Can be filtered by currency_pair and/or task_id.
    USE WITH CAUTION.
    """
    try:
        query = session.query(TradingTransaction)
        if currency_pair:
            query = query.filter(TradingTransaction.currency_pair == currency_pair)
        if task_id:
            query = query.filter(TradingTransaction.task_id == task_id)

        # To delete related MaxLost settings via cascade, if configured, otherwise delete them manually first or handle constraints.
        # Assuming cascade="all, delete-orphan" on TradingTransaction.max_lost_setting handles MaxLost.
        num_deleted = query.delete(synchronize_session=False) # 'fetch' or False
        session.commit() # Or commit can be handled by the caller context
        logger.warning(f"Deleted {num_deleted} trading transaction(s) for currency {currency_pair if currency_pair else 'all'}, task ID {task_id if task_id else 'all'}.")
        return num_deleted
    except Exception as e:
        session.rollback()
        logger.error(f"Error resetting trading transactions: {e}", exc_info=True)
        raise

# Placeholder for creating a new TradingTransaction (buy operation)
# The actual save_buy logic was on the SQLObject model instance.
# With SQLAlchemy, this would typically be:
# 1. Create instance: new_trade = TradingTransaction(...)
# 2. Add to session: session.add(new_trade)
# 3. Commit session: session.commit() (often done by the caller, e.g., in Trading.run)

# Example of how these might be used by `Trading.run()` (conceptual):
# def record_buy(session: Session, task_id: str, pair: str, total_value: float, fee: float, amount_bought: float, buy_price: float):
#     new_trade = TradingTransaction(
#         task_id=task_id,
#         currency_pair=pair,
#         buy_value_eur=total_value,
#         buy_fee_eur=fee,
#         base_currency_bought_amount=amount_bought,
#         base_currency_buy_price=buy_price,
#         buy_date_time=datetime.datetime.utcnow()
#     )
#     session.add(new_trade)
#     # session.commit() # Typically caller commits
#     return new_trade

# def record_sell(session: Session, trade_to_sell: TradingTransaction, sell_price: float, fee: float, profit: float):
#     trade_to_sell.sell_date_time = datetime.datetime.utcnow()
#     trade_to_sell.base_currency_sell_price = sell_price
#     trade_to_sell.sell_fee_eur = fee
#     trade_to_sell.profit_eur = profit
#     session.add(trade_to_sell) # Add to session if it was detached or to ensure it's marked dirty
#     # session.commit() # Typically caller commits
#     return trade_to_sell


def get_portfolio_value_history_sqlalchemy(session: Session, initial_capital: float):
    """
    Calculates the portfolio value history based on closed trades using SQLAlchemy.

    Args:
        session: The SQLAlchemy session to use for database queries.
        initial_capital: The initial capital to start the graph with.

    Returns:
        A list of (timestamp, portfolio_value) tuples.
        The first point is (earliest_buy_time, initial_capital).
        Subsequent points are (sell_time, initial_capital + cumulative_profit).
        Returns a list with a single point (now, initial_capital) if no trades exist.
    """
    data_points = []
    cumulative_profit = 0.0

    # Find the earliest trade time for the initial data point
    first_trade = session.query(TradingTransaction).order_by(TradingTransaction.buy_date_time).first()

    start_time = datetime.datetime.utcnow()  # Default if no trades
    if first_trade and first_trade.buy_date_time:
        start_time = first_trade.buy_date_time

    data_points.append((start_time, initial_capital))

    # Query completed trades, ordered by sell_date_time
    completed_trades = session.query(TradingTransaction).filter(
        TradingTransaction.sell_date_time != None
    ).order_by(TradingTransaction.sell_date_time).all()

    for trade in completed_trades:
        if trade.profit_eur is not None: # Assuming profit is stored in profit_eur
            cumulative_profit += trade.profit_eur
            if trade.sell_date_time: # Should always be true due to query condition
                data_points.append((trade.sell_date_time, initial_capital + cumulative_profit))
            else:
                # This case should ideally not be reached if data integrity is maintained
                logger.warning(f"Trade with ID {trade.id} marked complete but missing sell_date_time, skipping for graph.")

    # If there were no completed trades, the list will only contain the initial point.
    # If there were no trades at all, it will be (now, initial_capital)
    # If there were trades, but none completed, it will be (first_trade_buy_time, initial_capital)
    # This matches the behavior of the original function where if completed_trades is empty,
    # it just returns the initial data_points list.

    return data_points


def get_completed_trades_sqlalchemy(session: Session) -> list[TradingTransaction]:
    """
    Retrieves all completed trades (those with a sell_date_time) ordered by sell_date_time.

    Args:
        session: The SQLAlchemy session to use for database queries.

    Returns:
        A list of TradingTransaction objects representing completed trades.
    """
    try:
        completed_trades = session.query(TradingTransaction).filter(
            TradingTransaction.sell_date_time != None
        ).order_by(TradingTransaction.sell_date_time).all()
        logger.debug(f"Retrieved {len(completed_trades)} completed trades.")
        return completed_trades
    except Exception as e:
        logger.error(f"Error retrieving completed trades: {e}", exc_info=True)
        # Depending on desired error handling, could return [] or re-raise
        raise
