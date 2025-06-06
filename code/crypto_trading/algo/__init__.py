# crypto_trading/algo/__init__.py
from .algoIf import AlgoIf
from .average import GuppyMMA
from .bollinger import Bollinger
from .moving_average_crossover import MovingAverageCrossover

__all__ = ['AlgoIf', 'GuppyMMA', 'Bollinger', 'MovingAverageCrossover']
