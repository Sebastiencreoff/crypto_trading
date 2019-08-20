# -*- coding:utf-8 -*-

import pytest

import crypto_trading.connection as connection

TEST_DIR = os.path.realpath(os.path.dirname(__file__))
COINBASE_FILE = os.path.join(TEST_DIR, 'inputs', 'coinbase.json')

@pytest.fixture()
def coinbase_file(tmpdir):
    f = tmpdir.mkdir('config').join('coinbase.json')

@pytest.fixture()
def coinbase_factory(coinbase_file):
    return connection.coinBase.CoinBaseConnect(COINBASE_FILE)


@pytest.fixture()
def simu_factory():

    def create_simu(file=None, type=None):
        if type == 'random':
            return connection.simulation.SimulationConnect()
        elif type == 'file':
            return connection.simulation.SimulationConnect(, file)

    return create_simu
