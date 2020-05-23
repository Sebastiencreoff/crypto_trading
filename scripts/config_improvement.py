#! /usr/bin/env python
import copy
import json
import pathlib

from phoenics import Phoenics

from crypto_trading import trading


CONFIG = {
    "GuppyMMA" : {
        "short_term" : [30, 50, 80, 100, 120, 150],
        "long_term" : [300, 350, 400, 450, 500, 600],
        "buy" : 6,
        "sell" : 6
    },
    "maxLost" : {
        "percentage" : 3,
        "percentage_update"   : 0.25,
        "mean" : 720
    },
    "takeProfit": {
        "percentage" : 5
    },
    "Bollinger": {
        "frequency" : 100
    }
}

MAX_ITER = 100

def merit_function(params):
    # Update algo configuration.
    config = copy.deepcopy(CONFIG)

    config['GuppyMMA']['buy'] = int(params['GuppyMMA_buy'][0])
    config['GuppyMMA']['sell'] = int(params['GuppyMMA_sell'][0])
    config['maxLost']['percentage'] = int(params['maxLost_perc'][0])
    config['maxLost']['percentage_update'] = float(params['maxLost_perc_update'][0])
    config['maxLost']['mean'] = int(params['maxLost_mean'][0])
    config['takeProfit']['percentage'] = float(params['takeProfit_perc'][0])
    config['Bollinger']['frequency'] = int(params['bollinger'][0])

    pathlib.Path('./config/algo.json').write_text(json.dumps(config, indent=2))

    
    simu = trading.Trading('./config/trading_SIMU.json')
    try:
        simu.run()
    except IndexError as e:
        pass
    params['profit'] = simu.profits()
    simu.reset()
    print(f'Profit: {params["profit"]} Config: {config}')
    return params

def main():
    phoenics = Phoenics('scripts/phoenics.json')

    observations = []
    for num_iter in range(MAX_ITER):
	
        print(f'RUN: {num_iter}')
        # query for new parameters
        params = phoenics.recommend(observations = observations)
    
        # evaluate the proposed parameters
        for param in params:
            observation =  merit_function(param)


            observations.append(observation)

if __name__ == "__main__":
    main()