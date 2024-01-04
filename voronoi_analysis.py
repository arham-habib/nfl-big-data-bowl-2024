import pandas as pd
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, cKDTree
from concurrent.futures import ProcessPoolExecutor

# imports from the utils.py script
import utils as NFLUtils

# TODO: 
# define the address of 'games.csv'
# define the address of 'tracking_week_{week}.csv'

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
            if label % 5 == 0: 
                results = NFLUtils.analyze_game(game_id=game_id, tracking_file=f'./data/tracking_week_{week}.csv', animation=True)
            else: 
                results = NFLUtils.analyze_game(game_id=game_id, tracking_file=f'./data/tracking_week_{week}.csv', animation=False)
        except Exception as e: 
            print(e)
            continue

if __name__ == '__main__': 
    main()