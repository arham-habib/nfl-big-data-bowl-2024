import pandas as pd
import numpy as np
import time

import code.utils as NFLUtils


def main(): 
    games_file = './data/games.csv'
    games = pd.read_csv(games_file)

    for _, row in games.iterrows(): 
        try: 
            game_id = row.gameId
            week = row.week
            if game_id != 2022092500: #CHANGE ME!
                continue
            _ = NFLUtils.analyze_game(game_id=game_id, tracking_file=f'./data/tracking_week_{week}.csv', animation=True)
        except Exception as e: 
            print(e)
            continue
        
if __name__ == '__main__': 
    main()