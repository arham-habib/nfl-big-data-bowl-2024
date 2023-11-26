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

def main(): 

    games_file = './data/games.csv'
    games = pd.read_csv(games_file)
    games.head(10)

    start_time = time.time()
    for label, row in games.iloc[6:].iterrows(): 
        print("Runtime: ", time.time() - start_time)
        try: 
            print(row)
            game_id = row.gameId
            week = row.week
            results = NFLUtils.analyze_game(game_id=game_id, tracking_file=f'./data/tracking_week_{week}.csv')
            time.sleep(30)  # doing this to not melt my processor overnight
        except: 
            continue

if __name__ == '__main__': 
    main()