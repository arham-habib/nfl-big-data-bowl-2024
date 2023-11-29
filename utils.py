import pandas as pd
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpltPath
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, box, LineString, Point
from shapely.ops import unary_union
from scipy.spatial import Voronoi, cKDTree, voronoi_plot_2d, ConvexHull
from concurrent.futures import ProcessPoolExecutor, as_completed
import scipy as sp

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Organizational methods

def load_game_data(tracking_file_path: str, plays_file_path: str, game_id: int, chunk_size:int = 10000)->pd.DataFrame:
    """
    Load rows from a CSV file that match a specific gameID

    Args:
    file_path (str): Path to the CSV file
    plays_file_path (str): path to the plays CSV file
    game_id (int): the gameID to filter by
    chunk_size (int, optional): the number of rows per chunk, default 10000

    Returns:
    pd.DataFrame: a DataFrame containing rows with the specified gameID
    """
    data = pd.DataFrame()
    # stream data in chunks
    for chunk in pd.read_csv(tracking_file_path, chunksize=chunk_size):
        filtered_chunk = chunk[chunk['gameId'] == game_id]
        # when no more matches, don't parse the rest of the file
        if filtered_chunk.shape[0] == 0:
            continue
        data = pd.concat([data, filtered_chunk], ignore_index=True)
    plays_df = pd.read_csv(plays_file_path)
    data = pd.merge(data, plays_df.loc[:,['gameId', 'playId', 'possessionTeam', 'ballCarrierId']], on=['gameId', 'playId'])
    data = data.loc[data['club'] != 'football']
    data['is_offense'] = (data['possessionTeam'] == data['club'])
    return data

def organize_game_data(df: pd.DataFrame)->dict:
    """
    Organize game data into a nested dictionary structure.

    Args:
    df (pd.DataFrame): The DataFrame containing game data.

    Returns:
    dict: A nested dictionary with plays as keys and dictionaries of data where the key is the frame and the values are data from that frame
    """

    # Initialize the main dictionary
    game_dict = {}

    # Iterate over each unique play in the DataFrame
    for play_id in df['playId'].unique():

        play_df = df[df['playId'] == play_id]
        play_events = play_df['event'].unique()


        #for now, ignoring fumbles, but maybe later on we can count that as a tackle?
        if 'fumble' in play_events:
          continue
        
        play_df = play_df.copy()
        if play_df['playDirection'].iloc[0] == 'left':
          play_df['x'] = 120 - play_df['x']
          play_df['y'] = 53.3 - play_df['y']


        # Initialize the play's dictionary
        play_dict = {}

        start_frame = 1
        #another potentiall type of event to include is 'run', but for now i'm excluding that
        #because I'm not exactly sure what it means
        if 'pass_outcome_caught' in play_events:
          start_frame = play_df.loc[play_df['event'] == 'pass_outcome_caught']['frameId'].min()
        elif 'handoff' in play_events:
          start_frame = play_df.loc[play_df['event'] == 'handoff']['frameId'].min()
        else:
          continue

        #this limits us to plays where a tackle is made
        #not sure if we need special consideration for when a runner scores, so those plays are ignored for now
        #potentially could include 'out_of_bounds' and factor that into defensive play as well
        end_frame = 1
        if 'tackle' in play_events:
          end_frame = play_df.loc[play_df['event'] == 'tackle']['frameId'].min()
        else:
          continue

        # Iterate over each player in the play
        for frame_id in play_df['frameId'].unique():
            if (frame_id < start_frame) or (frame_id > end_frame):
              continue
            frame_df = play_df[play_df['frameId'] == frame_id]

            # Select and sort relevant columns
            columns = ['nflId', 'time', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'event', 'is_offense', 'ballCarrierId']
            frame_df = frame_df[columns]
            frame_df = frame_df.astype({'nflId': int, 'ballCarrierId': int})
            
            # Add the player's DataFrame to the play's dictionary
            play_dict[frame_id] = frame_df

        # Add the play's dictionary to the main dictionary
        game_dict[play_id] = play_dict

    return game_dict

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Visualization methods

def generate_color_map(nfl_ids):
    """
    Generates a color map for given NFL IDs.

    Parameters:
    - nfl_ids: List of unique NFL IDs.

    Returns:
    - Dictionary mapping each NFL ID to a color.
    """
    nfl_ids = nfl_ids.dropna().unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(nfl_ids)))
    color_map = {nfl_id: color for nfl_id, color in zip(nfl_ids, colors)}
    return color_map

def create_animation(frame_dict: dict, tpc_per_frame: dict, play_filepath:str, x_min=0, x_max=120, y_min=0, y_max=53.3, x_step=1, y_step=1):
    """
    Creates an animation of bucketed Voronoi spaces for different frames.

    Parameters:
    - frame_dict: Dictionary of DataFrames indexed by frame, each containing ['player_id', 'x', 'y'].
    - tpc_per_frame (dict): returned from the tackle_percentage_contribution_per_play method that labels the contribution of each defensive player per play, each key is the frame
    - play_filepath (str): the filepath used to save the animation 
    - min_x (float): the min x value in the graph (long side of football field, 0-120)
    - max_x (float): the max x
    - min_y (float): the min y value in the graph (short axis of football field, 0-53.3)
    - max_y (float): the max y
    - frame (int): the frame in question, useful for locating the file
    
    Returns:
    - None

    """
    # assign a color map for all players in the play, based on which players were active in the first frame
    color_map = generate_color_map(frame_dict[sorted(frame_dict.keys())[0]].nflId) # from the first frame, pull all active players

    # open plots were taking too much memory
    plt.close('all')

    # Function to draw a single frame for the animation
    def draw_frame(frame_number):
        
        # Process the frame data to get the assignments
        player_assignments = assign_squares_to_players(frame_dict[frame_number], x_min, x_max, y_min, y_max, x_step, y_step)
        ball_carrier = frame_dict[frame_number].ballCarrierId.iloc[0]

        nonlocal color_map
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Create a list to hold all the rectangles
        rectangles = []

        # Create a list to hold the colors of each rectangle
        rectangle_colors = []

        for _, row in player_assignments.iterrows():

            # plot the colors based on the closest player
            player_id = row['closest_player_id']
            square_color = color_map.get(player_id, 'grey')
            rect = patches.Rectangle((row['square_x'] - 0.5, row['square_y'] - 0.5), 1, 1)
            rectangles.append(rect)
            rectangle_colors.append(square_color)

        # Add labels at centroids
        player_positions = zip(frame_dict[frame_number].nflId, frame_dict[frame_number].is_offense, frame_dict[frame_number].x, frame_dict[frame_number].y)
        for player_id, is_offense, x, y in player_positions:

            # Get tackle percentage contribution, default 0 for offense
            tpc = tpc_per_frame[frame_number].get(player_id, 0) 

            # label the offensive players, red=ball carrier, black=offense, white=defense
            if player_id == ball_carrier: 
                dot_color='red'
            elif is_offense: 
                dot_color='black'
            else: 
                dot_color='white'

            # plot the dot for every player and their TPC
            ax.plot(x, y, marker='o', markersize=5, markerfacecolor=dot_color)
            ax.text(x, y, f'{player_id}: {tpc}', ha='center', va='center', fontsize=9)
        # Create a PatchCollection and add it to the axis
        pc = PatchCollection(rectangles, facecolor=rectangle_colors, edgecolor=None)
        ax.add_collection(pc)

        # Additional plot settings
        ax.set_xlabel('Yards (X-axis)')
        ax.set_ylabel('Yards (Y-axis)')
        ax.set_title(f'Bucketed Voronoi Areas (ball carrier: {ball_carrier})')

    # Create figure and axis for the animation
    fig, ax = plt.subplots(figsize=(24, 12))

    # Create the animation
    anim = FuncAnimation(fig, draw_frame, frames=sorted(frame_dict.keys()), interval=200, repeat=False)

    # To save the animation, uncomment the line below and specify the filename and writer
    anim.save(play_filepath + f'/voronoi_visualizer.mp4', writer='ffmpeg')

    # plt.show()
    # return anim

def visualize_field(frame:pd.DataFrame, x_min, color_map:dict=None, x_max=110, y_min=0, y_max=53.3, x_step=1, y_step=1):

    if not color_map: 
        color_map = generate_color_map(frame.nflId)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Process the frame data to get the assignments
    player_assignments = assign_squares_to_players(frame, x_min, x_max, y_min, y_max, x_step, y_step)
    ball_carrier = frame.ballCarrierId.iloc[0]

    ax.clear()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Create a list to hold all the rectangles
    rectangles = []

    # Create a list to hold the colors of each rectangle
    rectangle_colors = []

    for _, row in player_assignments.iterrows():

        # plot the colors based on the closest player
        player_id = row['closest_player_id']
        square_color = color_map.get(player_id, 'grey')
        rect = patches.Rectangle((row['square_x'] - 0.5, row['square_y'] - 0.5), 1, 1)
        rectangles.append(rect)
        rectangle_colors.append(square_color)

    # Add labels at centroids
    player_positions = zip(frame.nflId, frame.is_offense, frame.x, frame.y)
    for player_id, is_offense, x, y in player_positions:

        # Get tackle percentage contribution, default 0 for offense
        tpc = voronoi_area(player_assignments).get(player_id, 0) 

        # label the offensive players, red=ball carrier, black=offense, white=defense
        if player_id == ball_carrier: 
            dot_color='red'
        elif is_offense: 
            dot_color='black'
        else: 
            dot_color='white'

        # plot the dot for every player and their TPC
        ax.plot(x, y, marker='o', markersize=5, markerfacecolor=dot_color)
        ax.text(x, y, f'{player_id}: {tpc}', ha='center', va='center', fontsize=9)
    # Create a PatchCollection and add it to the axis
    pc = PatchCollection(rectangles, facecolor=rectangle_colors, edgecolor=None)
    ax.add_collection(pc)

    # Additional plot settings
    ax.set_xlabel('Yards (X-axis)')
    ax.set_ylabel('Yards (Y-axis)')
    ax.set_title(f'Bucketed Voronoi Areas (ball carrier: {ball_carrier})')

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Voronoi methods

def assign_squares_to_players(frame_data, x_min=0, x_max=120, y_min=0, y_max=53.3, x_step=1, y_step=1):
    """
    Assigns each x_step by y_step square of a football field to the nearest player using Voronoi tessellation.

    Parameters:
    - frame_data (pd.DataFrame): DataFrame with columns ['nflId', 'x', 'y'] representing players' positions.
    - x_min, x_max (float): Optional. The minimum and maximum x-coordinates (in yards) of the field area to consider.(0-120 yards)
    - y_min, y_max (float): Optional. The minimum and maximum y-coordinates (in yards) of the field area to consider.(0-53.3 yards)
    - x_step, y_step (float): Optional. The size of each Voronoi bucket, defaults to 1 yd by 1 yd

    Returns:
    - A DataFrame with columns ['square_x', 'square_y', 'closest_player_id', 'ball_carrier', 'is_offense'].
    """
    # modify the frame_data such that ever offensive player gets the ballCarrierId (assume they share voronoi space)
    # commenting this out for the moment because it wasn't helping the analysis, but in the future, make it such that if they're touching the space of another offensive player they become one unit
    ball_carrier = frame_data.ballCarrierId.iloc[0]
    # frame_data.loc[frame_data.is_offense == True, 'nflId'] = ball_carrier

    # Generate Voronoi diagram
    points = frame_data[['x', 'y']].values
    vor = Voronoi(points)
    # fig = voronoi_plot_2d(vor)
    # plt.show()  # for debug purposes 

    # Generate all 1-yard squares within specified limits
    x_range = np.arange(x_min, x_max + x_step, x_step)
    y_range = np.arange(y_min, y_max + y_step, y_step)
    squares = pd.DataFrame([(x, y) for x in x_range for y in y_range], columns=['square_x', 'square_y'])

    # Create a KDTree for efficient nearest neighbor search
    tree = cKDTree(points)

    # Assign each square to the closest player based on Voronoi regions
    squares['closest_player_id'] = squares.apply(lambda row: frame_data.iloc[tree.query((row['square_x'], row['square_y']))[1]]['nflId'], axis=1)
    # get the ID of the ball carrier
    squares['ball_carrier'] = ball_carrier

    return squares


def voronoi_area(squares: pd.DataFrame, weights: pd.DataFrame=None):
    """
    Return the area attributed to each unique player by nflID

    Params: 
    - squares (pd.DataFrame): a dataframe with columns ['square_x', 'square_y', 'closest_player_id'].
    - weights (pd.DataFrame): 

    Returns: 
    - a dictionary with keys of closest_player_id and values of the voronoi areas, in square yards (we can modify this later with the weights)
    """
    # this is the case where we weight each Voronoi bin differently -- we can implement this later
    if weights: 
        return 
    else:
        voronoi_areas = squares.groupby('closest_player_id').size().to_dict()
    
    return voronoi_areas

def in_box(players, bounding_box):
    """ 
    Check if a point is in a box (works in conjunction with calculate_voronoi_areas)
    Params: 
    - players (np.array()): a 2D array of the coordinates of all the players
    - bounding_box (tuple): the coordinates of the bounds in the form (x_min, x_max, y_min, y_max)
    """
    return np.logical_and(np.logical_and(bounding_box[0] <= players[:, 0],
                                         players[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= players[:, 1],
                                         players[:, 1] <= bounding_box[3]))

def calculate_voronoi_areas(df, x_min:float=None, x_max=110, y_min=0, y_max=53.3): 
    """
    Take 2: use mirroring to solve
    df (pd.DataFrame): the data frame of frame_data from the organize_game_data method
    x_min (float): the minimum x value at which we end analysis
    x_max (float): default 100 because that's the endzone, the amount of defended territory
    y_min (float), y_max(float): bounds for y
    """
    
    # create a boundary 10 yards behind the ball carrier or 10 yds (start of endzone), whichever is greater
    if not x_min: 
        x_min = max(df[df.nflId==df.ballCarrierId.iloc[0]].x.iloc[0] - 10, 10)

    df_filtered = df[df['x'].between(x_min, x_max) & df['y'].between(y_min, y_max)]
    players = df_filtered[['x', 'y']].to_numpy()

    bounding_box = (x_min, x_max, y_min, y_max)
    # Select towers inside the bounding box
    i = in_box(players, bounding_box)
    # Mirror points
    points_center = players[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = sp.spatial.Voronoi(points)
    # Filter regions
    regions = []
    [vor.point_region[i] for i in range(10)]

    vor.filtered_points = points_center
    vor.filtered_regions = [vor.regions[vor.point_region[i]] for i in range(len(points_center))]
    areas = [ConvexHull(vor.vertices[vor.filtered_regions[idx], :]).volume for idx in range(len(vor.filtered_regions))]  # pull the areas and zip them to the vertices passed in
    
    return dict(zip(df_filtered.nflId, areas))

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TPC Methods

def tackle_percentage_contribution_per_frame(frame_data:pd.DataFrame, x_step: int=1, y_step: int=1)->dict:
    """ 
    For every unique player attributed to a square on the defending team, take them out and see how much Voronoi area would be gained by the player in possession. 

    Params: 
    - frame_data (pd.DataFrame): a dataframe from the organize_game_data method with columns ['nflId', 'ballCarrierId', 'is_offense', 'x', 'y']
    - x_step (int): the x-side of the voronoi bins when caling the assign_squres_to_players method
    - y_step (int): the y-side of the voronoi bins when caling the assign_squres_to_players method

    Returns: 
    - dictionary with keys of nflId and value of the tackle percentage contribution for that frame
    """
    
    area_protected = {}
    # get the ball carrier and offensive players
    ball_carrier = frame_data.ballCarrierId.iloc[0]
    offensive_players = dict(zip(frame_data.nflId, frame_data.is_offense))

    # get the minimum x, after which we will cut off voronoi analysis
    x_min = max(10, frame_data.loc[frame_data.nflId==ball_carrier, 'x'].iloc[0] - 10) # we end the voronoi tesselation 10 yards behind the ball carrier or 10, whichever is greater
    baseline_area = calculate_voronoi_areas(frame_data, x_min=x_min).get(ball_carrier,0)
    # squares = assign_squares_to_players(frame_data, x_min=x_min, x_step=x_step, y_step=y_step)
    # baseline_area = voronoi_area(squares)[ball_carrier]
    
    for player_id in frame_data.nflId.unique(): 
        # break for the ball_carrier     
        if offensive_players[player_id]: 
            continue
        # take the frame data if that player didn't exist
        filtered_frame_data = frame_data[frame_data.nflId != player_id]
        protected_areas = calculate_voronoi_areas(filtered_frame_data, x_min=x_min).get(ball_carrier,0)
        # calculate how much additional space the offense gets
        # voronoi_filtered = assign_squares_to_players(filtered_frame_data, x_min=x_min, x_step=x_step, y_step=y_step)
        # protected_areas = voronoi_area(voronoi_filtered)[ball_carrier]
        area_protected[player_id] = protected_areas - baseline_area  # how much more area do they get?
    
    # divide by the total sum of the frame to get tackle percentage contribution in each frame
    # I'm unconvinced this is the correct approach and I'm commenting out out for now, we can talk about this
    # Basically, if no one is close to the player on offense, I think this will be misleading
    # total_protected_area = sum(area_protected.values())
    # for key, value in area_protected.items(): 
    #     area_protected[key] = value / total_protected_area

    return area_protected


def tackle_percentage_contribution_per_play(frame_dict:dict, filepath:str, x_step:int=1, y_step:int=1, animation:bool=False): 
    """
    This iterates through the frames in any given play and calculates the tackle percentage contribution of each player
    
    Params: 
    - frame_dict: dict from the organize_game_data method for each play
    - filepath (str): the filepath of the folder under which we are caching play data
    - x_step (int): the x-side of the voronoi bins when caling the assign_squres_to_players method
    - y_step (int): the y-side of the voronoi bins when caling the assign_squres_to_players method
    - animation (bool): whether or not we want to make an MP4 of the play

    Returns: 
    - dictionary with keys of nflId and value of the tackle percentage contribution for that play
    """
    # empty dict, one indexed by player, the other indexed by frame
    total_tpc = {}
    tpc_per_frame = {}

    # sort the frames
    frame_dict_sorted = sorted(frame_dict.items(), key=lambda x: x[0])
    # iterate through the frames of the play
    for key, frame in frame_dict_sorted: 

        # get protected areas, append to both dictionaries
        frame_tpc = tackle_percentage_contribution_per_frame(frame, x_step, y_step)
        tpc_per_frame[key] = frame_tpc

        # append to the overall dict for the play
        for player, contribution in frame_tpc.items():
            if player in total_tpc.keys(): 
                total_tpc[player] += contribution
            else: 
                total_tpc[player] = contribution
    
    # normalize every player's contribution such that it sums to 1
    total_protected_area = sum(total_tpc.values())
    for key, value in total_tpc.items():
        total_tpc[key] = value / total_protected_area


    # Convert the dictionary with the frame data to a DataFrame to cache
    # The keys of the outer dict become the index, and the inner dicts' keys become the column names
    tpc_per_frame_df = pd.DataFrame.from_dict(tpc_per_frame, orient='index')

    # Save to CSV, with the index to make future multiplication easier
    tpc_per_frame_df.to_csv(f'{filepath}/tpc_per_frame_updated_voronoi.csv', index=True)

    # cast everything to strings from int64 (otherwise cannot store in JSON)
    total_tpc_converted = {str(key): value for key, value in total_tpc.items()}

    # cache this result as a JSON for each play
    json.dump(total_tpc_converted, open(filepath+'/tpc_updated_voronoi.json', 'w'))

    # create an animation
    if animation: 
        create_animation(frame_dict=frame_dict, tpc_per_frame=tpc_per_frame, play_filepath=filepath, x_step=x_step, y_step=y_step)

    return total_tpc

def analyze_play(key, play, filepath, x_step, y_step, animation):
    """
    Wrapper function to analyze a single play. This function will be executed in parallel.
    Params: 
    - key (int): the play number
    - play (dict): a dictionary of the frames (pd.DataFrame) of each play
    - filepath (str): the filepath of tha game within which we will save information/animations of the play
    - x_step, y_step (float): the x/y step of each of the voronoi bins
    """
    print(key)  # For debugging purposes

    # Define the play's file path
    play_filepath = f'{filepath}/{key}'
    if not os.path.exists(play_filepath):
        os.makedirs(play_filepath)

    try:
        # Calculate the tackle_percentage_contribution
        # Ensure that the tackle_percentage_contribution_per_play function is defined appropriately
        play_tpc = tackle_percentage_contribution_per_play(frame_dict=play, filepath=play_filepath, x_step=x_step, y_step=y_step, animation=animation)

        return {player: contribution for player, contribution in play_tpc.items()}
    except Exception as e:
        print(f'Error processing play {key}: {e}')
        return {}

def analyze_game(game_id, tracking_file, x_step=1, y_step=1, plays_file='./data/plays.csv', players_file='./data/players.csv', game_file='./data/games.csv', animation:bool=False):
    """ 
    A method to analyze a game. Calling this will analyze and cache all the plays + the results of the analysis
    Param: 
    - game_id (int): the ID of the game as found in the Kaggle cleaned data
    - tracking_file (str): the address of the file in which the tracking data is stored
    - x_step, y_step (float): the x and y steps of each of the voronoi bins
    - plays_file (str): the address of the plays file
    - players_file (str): the filepath of the file containing players [TODO: CURRENTLY UNUSED.]
    - game_file (str): the filepath of the file containing information about each game
    """
    
    games = pd.read_csv(game_file)
    game_data = games[games.gameId==game_id].iloc[0, [0, 5, 6]] # pull the ID (col 0), home team (col 5), visitng team (col 6)
    filepath = f'./games/{game_data.iloc[0]}_{game_data.iloc[1]}_{game_data.iloc[2]}'

    # Create a directory for the game if none exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Sort and organize the data
    game_data_organized = organize_game_data(load_game_data(tracking_file, plays_file, game_id))
    sorted_game_data_organized = sorted(game_data_organized.items(), key=lambda x: x[0])

    # Dictionary to store the overall tackle_percentage_contribution
    game_tpc = {}

    # Using ProcessPoolExecutor to parallelize the loop
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_play, key, play, filepath, x_step, y_step, animation) for key, play in sorted_game_data_organized]

        for future in as_completed(futures):
            play_tpc = future.result()
            for player, contribution in play_tpc.items():
                game_tpc[player] = game_tpc.get(player, 0) + contribution

    # Convert game_tpc keys from int64 to string to store in JSON
    game_tpc_converted = {str(key): value for key, value in game_tpc.items()}

    # Cache this result as a JSON for each game
    json.dump(game_tpc_converted, open(filepath + '/game_tpc.json', 'w'))

    return game_tpc

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Euclidean Distance utils

def euclidean_distance_per_frame(frame_data:pd.DataFrame)->dict: 
    """ 
    Params: 
    - frame_data (pd.DataFrame): a dataframe from the organize_game_data method with columns ['nflId', 'ballCarrierId', 'is_offense', 'x', 'y']
    Returns: 
    - distance_dict (dict): a dict where the keys are the player IDs and the values are the distances
    """
    ball_carrier = frame_data.ballCarrierId.iloc[0]
    x, y = frame_data[frame_data.nflId==ball_carrier][['x', 'y']].iloc[0]
    defense = frame_data[~frame_data.is_offense].nflId
    distances = [np.sqrt((x-x_d)**2 + (y-y_d)**2) for x_d, y_d in zip(frame_data[~frame_data.is_offense].x, frame_data[~frame_data.is_offense].y)]
    distance_dict = dict(zip(defense, distances))

    return distance_dict

def euclidean_distance_per_play(frame_dict:dict, filepath:str)->dict: 
    """ 
    Calculate the Euclidean distances of each of the defenders from the ball
    Params: 
    - frame_dict: dict from the organize_game_data method for each play
    - filepath: the path of each play, under which we can cache the data
    """
    # Create a directory for the game if none exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    frame_distances = {}
    
    # sort the frames
    frame_dict_sorted = sorted(frame_dict.items(), key=lambda x: x[0])
    # iterate through the frames of the play
    for key, frame in frame_dict_sorted: 
        frame_distances[key] = euclidean_distance_per_frame(frame)

    # Convert the dictionary with the frame data to a DataFrame to cache
    # The keys of the outer dict become the index, and the inner dicts' keys become the column names
    frame_distances_df = pd.DataFrame.from_dict(frame_distances, orient='index')

    # Save to CSV, with the index to make future multiplication easier
    frame_distances_df.to_csv(f'{filepath}/distances_per_frame.csv', index=True)
    
    return frame_distances

def analyze_game_distances(game_id, tracking_file, plays_file='./data/plays.csv', game_file='./data/games.csv')->None:
    """ 
    A method to cache the distances of the players from the ball at all times
    Params: 
    - game_id (int): the ID of the game as found in the Kaggle cleaned data
    - tracking_file (str): the address of the file in which the tracking data is stored
    - plays_file (str): the address of the plays file
    - game_file (str): the filepath of the file containing information about each game
    """
    # read information about all the games
    games = pd.read_csv(game_file)
    game_data = games[games.gameId==game_id].iloc[0, [0, 5, 6]] # pull the ID (col 0), home team (col 5), visitng team (col 6)
    filepath = f'./games/{game_data.iloc[0]}_{game_data.iloc[1]}_{game_data.iloc[2]}'

    # Create a directory for the game if none exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Sort and organize the data
    game_data_organized = organize_game_data(load_game_data(tracking_file, plays_file, game_id))
    sorted_game_data_organized = sorted(game_data_organized.items(), key=lambda x: x[0])

    # Using ProcessPoolExecutor to parallelize the loop
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(euclidean_distance_per_play, play, f'{filepath}/{key}') for key, play in sorted_game_data_organized]
