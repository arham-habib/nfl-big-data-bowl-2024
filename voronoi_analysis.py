import pandas as pd
import numpy as np
import time
import os
# import json
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.collections import PatchCollection
# from matplotlib.animation import FuncAnimation
# from scipy.spatial import Voronoi, cKDTree
# from concurrent.futures import ProcessPoolExecutor

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
            results = NFLUtils.analyze_game(game_id=game_id, tracking_file=f'./data/tracking_week_{week}.csv', animation=False)
        except Exception as e: 
            print(e)
            continue

    # ######## This iterates through the tpc files, normalizes each frame to sum to one, and multiplies each row by the expected yards_{frame+1} - expected_yards_{frame}
    # # read in expected yards data
    # expected_yards = pd.read_csv('./data/eval_frame_df.csv')
    # # Define the path to the 'games' folder
    # games_folder = './games'

    # # Iterate over each game folder
    # for game_folder in os.listdir(games_folder):
    #     game_path = os.path.join(games_folder, game_folder)
    #     if os.path.isdir(game_path):
    #         # Extract gameId from the folder name
    #         gameId = int(game_folder.split('_')[0])
    #         print(gameId)

    #         # Iterate over each play folder within the game folder
    #         for play_folder in os.listdir(game_path):
    #             play_path = os.path.join(game_path, play_folder)
    #             if os.path.isdir(play_path):
    #                 # Extract playId from the folder name
    #                 playId = int(play_folder)

    #                 # Define the path to the tpc_per_frame_unweighted.csv file
    #                 tpc_file_path = os.path.join(play_path, 'tpc_per_frame_weighted_blockers.csv')
    #                 if os.path.exists(tpc_file_path):
    #                     # Load the tpc_per_frame_unweighted DataFrame
    #                     tpc_df = pd.read_csv(tpc_file_path, index_col=0)

    #                     # Normalize each row so that it sums to 1
    #                     tpc_df = tpc_df.div(tpc_df.sum(axis=1), axis=0)

    #                     # Filter the expected_yards DataFrame for the current gameId and playId
    #                     filtered_expected_yards = expected_yards[(expected_yards['gameId'] == gameId) & 
    #                                                             (expected_yards['playId'] == playId)]

    #                     # Initialize a DataFrame to store the results
    #                     constrictive_presence_ratio_df = pd.DataFrame(index=tpc_df.index, columns=tpc_df.columns)

    #                     # Iterate through the rows
    #                     for frame in tpc_df.index:
    #                         # Ensure the frame exists in the filtered_expected_yards
    #                         if frame in filtered_expected_yards['frameId'].values:
    #                             # Calculate the difference in expected yards for each frame
    #                             current_frame_row = filtered_expected_yards[filtered_expected_yards['frameId'] == frame]
    #                             next_frame_row = filtered_expected_yards[filtered_expected_yards['frameId'] == frame + 1]
    #                             if not next_frame_row.empty:
    #                                 delta_expected_yards = current_frame_row['expectedRemainingYardsByCarrier'].values[0]- next_frame_row['expectedRemainingYardsByCarrier'].values[0]
    #                                 # Multiply the percentages for each player by the delta
    #                                 # print(tpc_df.loc[frame])
    #                                 # print(delta_expected_yards)
    #                                 # print(tpc_df.loc[frame] * delta_expected_yards)
    #                                 constrictive_presence_ratio_df.loc[frame,:] = tpc_df.loc[frame,:] * delta_expected_yards

    #                     # Save the resulting DataFrame
    #                     # print(constrictive_presence_ratio_df)
    #                     constrictive_presence_ratio_df.to_csv(os.path.join(play_path, 'constrictive_presence_ratio_weighted_blockers.csv'))
    #     # augment with additional data like position and name

    # # Load the players DataFrame
    # players_file = './data/players.csv'
    # players_df = pd.read_csv(players_file)

    # # Load the constrictive presence ratio DataFrame
    # constrictive_file = './data/constrictive_presence_ratio_weighted_blockers.csv'
    # constrictive_df = pd.read_csv(constrictive_file)

    # # Merge the DataFrames on nflId
    # augmented_df = pd.merge(constrictive_df, players_df[['nflId', 'position', 'displayName']], 
    #                         on='nflId', how='left')

    # # Save the augmented DataFrame
    # augmented_df.to_csv('./data/constrictive_presence_ratio_weighted_blockers.csv', index=False)

    # print("Augmentation complete. Data saved to './data/constrictive_presence_ratio_weighted_blockers.csv'")

    # # Load the constrictive_presence_ratio_all DataFrame
    # constrictive_presence_ratio_all = pd.read_csv('./data/constrictive_presence_ratio_weighted_blockers.csv')

    # # Group by 'displayName' and sum 'constrictivePresenceSum'
    # grouped_by_player = constrictive_presence_ratio_all[['displayName', 'constrictivePresenceSum']].groupby('displayName').sum()

    # # Count the number of occurrences for each 'displayName'
    # counts = constrictive_presence_ratio_all['displayName'].value_counts()
    # std_player = constrictive_presence_ratio_all[['displayName', 'constrictivePresenceSum']].groupby('displayName').std()

    # # Divide the summed 'constrictivePresenceSum' by the count for each player
    # grouped_by_player['counts'] = counts # - std_player.constrictivePresenceSum / counts
    # grouped_by_player['std'] = std_player

    # grouped_by_player.to_csv('./data/cpr_per_player_weighted_blockers.csv', index=True)

if __name__ == '__main__': 
    main()