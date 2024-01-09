# GETS THE CPR FOR EVERY FRAME OF EVERY (VALID) PLAY

import pandas as pd
import numpy as np
import time

import code.utils as NFLUtils

def main(): 
    games_file = './data/games.csv'
    games = pd.read_csv(games_file)

    start_time = time.time()
    for label, row in games.iterrows(): 
        print(f"{label} ---- cumulative runtime: ", time.time() - start_time)
        try: 
            print(row)
            game_id = row.gameId
            week = row.week
            _ = NFLUtils.analyze_game(game_id=game_id, tracking_file=f'./data/tracking_week_{week}.csv', animation=False)
        except Exception as e: 
            print(e)
            continue
        
if __name__ == '__main__': 
    main()