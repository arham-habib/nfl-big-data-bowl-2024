import pandas as pd
import numpy as np
import scipy as sp
import os
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# ORGANIZATIONAL METHODS: USED TO READ IN THE GAMES

def load_game_data(tracking_file_path: str, plays_file_path: str, game_id: int, chunk_size:int = 10000)->pd.DataFrame:
    """
    Load rows from a CSV file that match a specific gameID

    Args:
    - file_path (str): Path to the CSV file
    - plays_file_path (str): path to the plays CSV file
    - game_id (int): the gameID to filter by
    - chunk_size (int, optional): the number of rows per chunk, default 10000

    Returns:
    - pd.DataFrame: a DataFrame containing rows with the specified gameID
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
    - df (pd.DataFrame): The DataFrame containing game data.

    Returns:
    - dict: A nested dictionary with plays as keys and dictionaries of data where the key is the frame and the values are data from that frame
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
          play_df['y'] = 160/3 - play_df['y']


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

def organize_game_data_eval_df(df: pd.DataFrame, valid_plays=None)->dict:
    """
    Organize game data into a nested dictionary structure. This method only reads in the methods in the eval_df used in the expected yards model. 

    Args:
    - df (pd.DataFrame): The DataFrame containing game data.

    Returns:
    - dict: A nested dictionary with plays as keys and dictionaries of data where the key is the frame and the values are data from that frame
    """

    # Initialize the main dictionary
    game_dict = {}

    # iterate through valid plays
    for play_id in valid_plays.playId.unique():
        
        # copy the data so that it can be transformed without inplace modification errors
        play_df = df[df['playId'] == play_id].copy()

        # Find the row corresponding to the given play_id
        play_row = valid_plays[valid_plays['playId'] == play_id]

        # Check if any rows match the play_id
        if not play_row.empty:
            # Extract the 'frameId_start' and 'frameId_end' values
            start_frame = play_row['frameId_start'].values[0]
            end_frame = play_row['frameId_end'].values[0]
        else:
            # Handle the case where no matching play_id was found
            start_frame = None
            end_frame = None

        if play_df['playDirection'].iloc[0] == 'left':
            play_df['x'] = 120 - play_df['x']
            play_df['y'] = 160/3 - play_df['y']

        # clip the coordinates
        play_df['x']= np.clip(play_df['x'], 0, 120)
        play_df['y'] = np.clip(play_df['y'], 0, 160/3)

        # create the dictionary to store information from the play
        play_dict = {}

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
# Voronoi methods

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

def calculate_voronoi_areas(df, x_min:float=0, x_max=120, y_min=0, y_max=160/3, plot_graph:bool=False, tpc_dict:dict=None, ax=None): 
    """
    Custom Voronoi utils. TLDR: mirror the points over x_min, x_max, y_min, y_max to create a boundaries. Use shoelace method to calculate area of each region. Some become negative because of quirks w the floating point numbers
    
    df (pd.DataFrame): the data frame of frame_data from the organize_game_data method. Has columsn: [nflId	time, playDirection, x, y, s, a, dis, o, dir, event, is_offense, ballCarrierId]
    x_min (float): the minimum x value at which we end analysis
    x_max (float): max x value in the analysis, default 110 because that's the endzone
    y_min (float), y_max(float): bounds for y
    plot_graph (bool): whether you want to generate a plot or not
    tpc_dict (dict): a dictionary indexed by nflId with the TPC for that frame of every player. Used in the plot_graph if specified. 
    ax: if calling from another method, pass the axis
    """
    
    # # create a boundary 10 yards behind the ball carrier or 10 yds (start of endzone), whichever is greater
    # if not x_min: 
    #     x_min = max(df[df.nflId==df.ballCarrierId.iloc[0]].x.iloc[0] - 10, 10)

    # filter points to the ones in the relevant region
    df_filtered = df[df['x'].between(x_min, x_max) & df['y'].between(y_min, y_max)].copy() # this seems redundant, but we need the df to be filtered to match each point to an nflId in the future. 
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

    # only pay attention to the points related to players in the relevant region
    vor.filtered_points = points_center
    vor.filtered_regions = [vor.regions[vor.point_region[i]] for i in range(len(points_center))]
    vertices = [vor.vertices[vor.filtered_regions[idx], :] for idx in range(len(vor.filtered_regions))]
    areas = [ConvexHull(vor.vertices[vor.filtered_regions[idx], :]).volume for idx in range(len(vor.filtered_regions))]  # pull the areas and zip them indexed to the vertices passed in
    df_filtered['voronoi_area'] = areas
    df_filtered['vertices'] = vertices

    # optionally, plot the graph
    if plot_graph:

        # Plot Voronoi diagram
        if not ax: # if there is no tpc_dict, this method is being called by itself and not with the helper animation() within tackle_percentage_contribution_per_play(), so we create a figure and plot
            fig, ax = plt.subplots(figsize=(24,16))

        # clear whatever was on the axis before
        ax.clear()
        
        # Plot boundaries
        # ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='black'))

        for i, region in enumerate(vor.filtered_regions):
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), fill=False, color='white', edgecolor='black')

            # Label points with nflId (smaller font size)
            nfl_id = df_filtered.iloc[i]['nflId']
            if tpc_dict: 
                player_tpc = tpc_dict.get(nfl_id, "") # try getting the TPC, and if not found, return a null value
            else:  # if no dict specified, no player has TPC
                player_tpc=""
            ax.text(vor.filtered_points[i][0], vor.filtered_points[i][1]-.75, f'{str(nfl_id)}: {player_tpc}', fontsize=10, ha='center', va='center') # the -.75 is to offset the points

            # Color points based on conditions
            if nfl_id == df_filtered.iloc[i]['ballCarrierId']:
                ax.plot(vor.filtered_points[i][0], vor.filtered_points[i][1], 'ro', markersize=10, label='Ball Carrier')
            elif nfl_id in df_filtered[df_filtered['is_offense']]['nflId'].tolist():
                ax.plot(vor.filtered_points[i][0], vor.filtered_points[i][1], 'mo', markersize=10, label='Offense')
            else:
                ax.plot(vor.filtered_points[i][0], vor.filtered_points[i][1], 'go', markersize=10, label='Defense')

        ballCarrierId = df_filtered[df_filtered['nflId'] == df_filtered.iloc[0]['ballCarrierId']]['ballCarrierId'].values[0]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.gca().set_aspect('equal', adjustable='box')
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_title(f'Voronoi Diagram (BallCarrierId: {ballCarrierId})')
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Ball Carrier', markerfacecolor='r', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='Offense', markerfacecolor='m', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='Defense', markerfacecolor='g', markersize=10)]
        ax.legend(handles=legend_elements)
    
    return df_filtered

# ---------------------------------------------------------------------------------------------------------------------------------------
# Blocker methods

def ccw(a, b, c)->bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def line_intersects_segment(p1: list[float, float], p2: list[float, float], 
                            q1: list[float, float], q2: list[float, float]) -> bool:
    """
    Check if the line segment p1-p2 intersects with the line segment q1-q2.
    """
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2))

def find_clear_path(df:pd.DataFrame)->list:
    """
    Find a player's position that has a clear path from the ball carrier.
    If a clear path exists, return the index of that player's position. 
    If not, return -1.
    """
    # empty list to hold blockers
    blockers = []

    # record the ball carrier's vertices
    ball_carrier_row = df[df['nflId'] == df['ballCarrierId']].iloc[0]
    x_0, y_0 = ball_carrier_row[['x', 'y']]
    # record the offensive player positions
    player_positions = df[df.is_offense][['nflId', 'x','y']]
    # record the voronoi regions of the defensive players
    voronoi_regions = df[~df.is_offense]['vertices']
    
    # iterate over the positions of the offensive players that are potentially blockers
    for _, id, x, y in player_positions.itertuples():
        # break for the ball carrier 
        if id == ball_carrier_row.nflId:
            continue
        path_clear = True
        for polygon in voronoi_regions:
            for j in range(len(polygon)):
                # Check if the line intersects with any of the polygon's edges
                if line_intersects_segment((x_0, y_0), (x,y), polygon[j], polygon[(j + 1) % len(polygon)]):
                    path_clear = False
                    break
            if not path_clear:
                break
        
        if path_clear:
            blockers.append(id)  # Clear path found to this player

    return blockers

def recognize_blockers(df:pd.DataFrame):
    blockers = find_clear_path(df)
    additional_area = df[df['nflId'].isin(blockers)]['weighted_voronoi_area'].sum()
    # Update the ball carrier's voronoi area
    df.loc[df['nflId'] == df['ballCarrierId'], 'weighted_voronoi_area'] += additional_area
    return df  # , blockers

# ----------------------------------------------------------------------------------------------------------------------------------
# Weighted Area Methods

def sort_vertices_clockwise(vertices):
    # Calculate the centroid of the polygon
    centroid = np.mean(vertices, axis=0)

    # Calculate the angle each vertex makes with the centroid
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])

    # Sort vertices based on angles
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]

    return sorted_vertices

def point_in_polygon_vectorized(x, y, vertices):
    N = len(vertices)
    inside = np.zeros(x.shape, dtype=bool)

    x_vertices, y_vertices = vertices[:, 0], vertices[:, 1]
    for i in range(N):
        p1_x, p1_y = x_vertices[i % N], y_vertices[i % N]
        p2_x, p2_y = x_vertices[(i + 1) % N], y_vertices[(i + 1) % N]

        conditions = np.logical_and(
            y > np.minimum(p1_y, p2_y),
            y <= np.maximum(p1_y, p2_y)
        )
        conditions = np.logical_and(
            conditions,
            x <= np.maximum(p1_x, p2_x)
        )

        if p1_y != p2_y:
            xinters = (y - p1_y) * (p2_x - p1_x) / (p2_y - p1_y) + p1_x
            conditions = np.logical_and(
                conditions,
                np.logical_or(p1_x == p2_x, x <= xinters)
            )

        inside = np.logical_xor(inside, conditions)

    # inside[:] = True

    return inside

# def point_in_polygon(x, y, vertices):
#     counter = 0
#     p1 = vertices[0]
#     N = len(vertices)
#     for i in range(1, N + 1):
#         p2 = vertices[i % N]
#         if y > min(p1[1], p2[1]):
#             if y <= max(p1[1], p2[1]):
#                 if x <= max(p1[0], p2[0]):
#                     if p1[1] != p2[1]:
#                         xinters = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
#                         if p1[0] == p2[0] or x <= xinters:
#                             counter += 1
#         p1 = p2
#     return not (counter % 2 == 0)

def angle_from_vector_vectorized(x_0, y_0, velocity_x, velocity_y, x_vals, y_vals):
    vector_to_point = np.array([x_vals - x_0, y_vals - y_0])
    velocity_vector = np.array([velocity_x, velocity_y])

    # Calculating magnitudes
    magnitude_vector_to_point = np.linalg.norm(vector_to_point, axis=0)
    magnitude_velocity_vector = np.linalg.norm(velocity_vector)

    # Element-wise multiplication and summing for dot product
    dot_product = np.sum(velocity_vector[:, np.newaxis, np.newaxis] * vector_to_point, axis=0)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        cosine_theta = np.clip(dot_product / (magnitude_velocity_vector * magnitude_vector_to_point), -1, 1)

    # Calculate angles
    theta_radians = np.arccos(cosine_theta)
    theta_degrees = np.degrees(theta_radians)

    # Setting angles to zero where either vector has zero magnitude
    theta_degrees = np.where((magnitude_velocity_vector * magnitude_vector_to_point) == 0, 0, theta_degrees)

    return theta_degrees

# def angle_from_vector(x_0, y_0, velocity_x, velocity_y, x, y):
#     vector_to_point = np.array([x - x_0, y - y_0])
#     velocity_vector = np.array([velocity_x, velocity_y])
#     magnitude_vector_to_point = np.linalg.norm(vector_to_point)
#     magnitude_velocity_vector = np.linalg.norm(velocity_vector)
#     # if the players aren't moving, return 0 as the angle
#     if magnitude_velocity_vector * magnitude_vector_to_point==0:
#         return 0
#     dot_product = np.dot(velocity_vector, vector_to_point)
#     cosine_theta = np.clip(dot_product / (magnitude_velocity_vector * magnitude_vector_to_point), -1, 1)
#     theta_radians = np.arccos(cosine_theta)
#     # Add more variables as needed
#     theta_degrees = np.degrees(theta_radians)
#     return theta_degrees

def weight_space_vectorized(x_0, y_0, velocity_x, velocity_y, x_vals, y_vals, max_x=120, max_y=160/3):
    # Vectorized computation of angles
    angle_from_velocity = angle_from_vector_vectorized(x_0, y_0, velocity_x, velocity_y, x_vals, y_vals)
    angle_from_endzone = angle_from_vector_vectorized(x_0, y_0, max_x - 10 - x_0, (max_y / 2) - y_0, x_vals, y_vals)

    # Vectorized computation of distance and speed
    distance = np.sqrt((x_vals - x_0)**2 + (y_vals - y_0)**2)
    speed = np.sqrt(velocity_x**2 + velocity_y**2)

    # Applying the formula in a vectorized way
    penalty = (angle_from_velocity + angle_from_endzone) / (360 * 4)
    weight = 1 / (0.5 + np.sqrt(distance)) - penalty

    return weight

# def weight_space(x_0,y_0, velocity_x, velocity_y, x, y, max_x=120, max_y=53.3):
#     angle_from_velocity = angle_from_vector(x_0, y_0, velocity_x, velocity_y, x, y)
#     angle_from_endzone = angle_from_vector(x_0, y_0, max_x - 10 - x_0, (max_y / 2) - y_0, x, y) # minus endzone length

#     distance = np.sqrt((x - x_0)**2 + (y - y_0)**2)
#     speed = np.sqrt((velocity_x**2) + (velocity_y**2))
#     penalty = (angle_from_velocity + angle_from_endzone) / (360 * 4)
#     weight = 1 / (0.5 + distance**0.5) - penalty
#     return weight

## THIS IS THE UPDATED VERSION 
# def calculate_Z_vectorized(x, y, dir, speed, num_ticks_per_yard=5, endzone_length=10):
#     max_x = 120
#     max_y = 160/3
#     velocity_x = speed * np.sin(np.radians(dir))
#     velocity_y = speed * np.cos(np.radians(dir))

#     # Vectorized generation of x_range and y_range
#     x_range = np.linspace(0, max_x, int(max_x * num_ticks_per_yard), endpoint=False)
#     y_range = np.linspace(0, max_y, int(max_y * num_ticks_per_yard), endpoint=False)

#     # Creating meshgrid for x and y values
#     x_mesh, y_mesh = np.meshgrid(x_range, y_range, indexing='ij')

#     # Vectorized weight_penalty and weight_distance calculations
#     Z_p = weight_penalty_vectorized(x, y, velocity_x, velocity_y, x_mesh, y_mesh, max_x, max_y, endzone_length)
#     Z_d = weight_distance_vectorized(x, y, x_mesh, y_mesh)

#     min_value_p = np.min(Z_p)
#     max_value_p = np.max(Z_p)
#     # Rescale the matrix to be between 0 and 1 (min-max scaling)
#     Z_p_scaled = (Z_p - min_value_p) / (max_value_p - min_value_p)

#     min_value_d = np.min(Z_d)
#     max_value_d = np.max(Z_d)
#     # Rescale the matrix to be between 0 and 1 (min-max scaling)
#     Z_d_scaled = (Z_d - min_value_d) / (max_value_d - min_value_d)

#     Z_scaled = Z_p_scaled * Z_d_scaled
#     # Vectorized operation to set values to 0 for x_val >= 110
#     Z_scaled[x_mesh >= 110] = 0

#     return Z_scaled

def weight_penalty_vectorized(x_0, y_0, velocity_x, velocity_y, x_vals, y_vals, max_x, max_y, endzone_length):
    angle_from_velocity = angle_from_vector_vectorized(x_0, y_0, velocity_x, velocity_y, x_vals, y_vals)
    angle_from_endzone = angle_from_vector_vectorized(x_0, y_0, max_x - endzone_length - x_0, (max_y / 2) - y_0, x_vals, y_vals)
    penalty = (angle_from_velocity + angle_from_endzone)**0.5
    return -penalty

def weight_distance_vectorized(x_0, y_0, x_vals, y_vals):
    distance = np.sqrt((x_vals - x_0)**2 + (y_vals - y_0)**2)
    weight = 1 / (0.5 + distance) # 1 / (0.5 + distance**0.5)
    return weight


def calculate_Z_vectorized(x_0, y_0, dir, speed, num_ticks_per_yard=5):
    max_x = 120
    max_y = 160/3
    velocity_x = speed * np.sin(np.radians(dir))
    velocity_y = speed * np.cos(np.radians(dir))

    # Vectorized generation of x_range and y_range
    x_range = np.linspace(0, max_x, int(max_x * num_ticks_per_yard), endpoint=False)
    y_range = np.linspace(0, max_y, int(max_y * num_ticks_per_yard), endpoint=False)

    # Creating meshgrid for x and y values
    x_mesh, y_mesh = np.meshgrid(x_range, y_range, indexing='ij')

    # Adjust for the center of each cell and compute weights
    Z = weight_space_vectorized(x_0, y_0, velocity_x, velocity_y, 
                                x_mesh + 1 / (2 * num_ticks_per_yard), 
                                y_mesh + 1 / (2 * num_ticks_per_yard)) * (1 / num_ticks_per_yard)**2

    # Normalize Z
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    # Vectorized operation to set values to 0 for x_val >= 110
    Z[x_mesh >= 110] = 0

    return Z

# def calculate_Z(x_0, y_0, dir, speed, num_ticks_per_yard=3):
#     max_x = 120
#     max_y = 53.3
#     velocity_x = speed * np.sin(np.radians(dir))
#     velocity_y = speed * np.cos(np.radians(dir))

#     x_range = [round(0 + i * (1 / (num_ticks_per_yard)), 4) for i in range(0, int(max_x * num_ticks_per_yard))]
#     y_range = [round(0 + i * (1 / (num_ticks_per_yard)), 4) for i in range(0, int(max_y * num_ticks_per_yard))]

#     Z = np.zeros((len(x_range), len(y_range)))

#     for i, x_val in enumerate(x_range):
#         for j, y_val in enumerate(y_range):
#             Z[i, j] = weight_space(x_0, y_0, velocity_x, velocity_y,
#                                 x_val + (1 / (2 * num_ticks_per_yard)),
#                                 y_val + (1 / (2 * num_ticks_per_yard))) * (1 / num_ticks_per_yard)**2
            
#     min_value = np.min(Z)
#     max_value = np.max(Z)
#     Z = (Z - min_value) / (max_value - min_value)

#     for i, x_val in enumerate(x_range):
#         for j, y_val in enumerate(y_range): 
#             if x_val >= 110: 
#                 Z[i,j] = 0

#     return Z

def calculate_weighted_area(vertices_col, Z, num_ticks_per_yard=5, vertices_to_area=None):
    max_x = 120
    max_y = 160/3
    result = []

    x_range = np.linspace(0, max_x, int(max_x * num_ticks_per_yard), endpoint=False)
    y_range = np.linspace(0, max_y, int(max_y * num_ticks_per_yard), endpoint=False)
    x_range_delta, y_range_delta = x_range[1] - x_range[0], y_range[1] - y_range[0]

    for vertices in vertices_col:
        vertices = sort_vertices_clockwise(vertices)
        flat_vertices = tuple(vertices.flatten())

        # Use cached result if available
        if flat_vertices in vertices_to_area:
            result.append(vertices_to_area[flat_vertices])
            continue

        # Calculate bounding box of the polygon
        bounding_min_x = next((x for x in reversed(x_range) if x <= np.min(vertices[:, 0]) - 1), 0)
        bounding_max_x = next((x for x in x_range if x >= np.max(vertices[:, 0]) + 1), x_range[-1])
        bounding_min_y = next((y for y in reversed(y_range) if y <= np.min(vertices[:, 1]) - 1), 0)
        bounding_max_y = next((y for y in y_range if y >= np.max(vertices[:, 1]) + 1), y_range[-1])

        # print(vertices)
        # print(bounding_min_x, bounding_max_x)
        # print(bounding_min_y, bounding_max_y)

        # Generate a grid of points within the bounding box
        # Add x_range_delta / 2 and y_range_delta / 2 to consider center points of the boxes
        x_vals = np.linspace(bounding_min_x + x_range_delta / 2, bounding_max_x + x_range_delta / 2, int((bounding_max_x - bounding_min_x) * num_ticks_per_yard) + 1, endpoint=True, dtype='float32') # [i + bounding_min_x for i in range(bounding_max_x - bounding_min_x)] # [x / 2.0 for x in range(241)
        y_vals = np.linspace(bounding_min_y + y_range_delta / 2, bounding_max_y + y_range_delta / 2, int((bounding_max_y - bounding_min_y) * num_ticks_per_yard) + 1, endpoint=True, dtype='float32') # [i + bounding_min_y for i in range(bounding_max_y - bounding_min_y)] # [y / 2.0 for y in range(107)

        # print('x and y vals')
        # print(x_vals)
        # print(y_vals)

        if len(x_vals) == 0 or len(y_vals) == 0:
            result.append(0)
            continue

        grid_x, grid_y = np.meshgrid(x_vals, y_vals, indexing='ij')

        # Check which points are inside the polygon
        mask = point_in_polygon_vectorized(grid_x, grid_y, vertices)

        # Map grid coordinates to Z array indices
        grid_x_indices = ((grid_x - x_range_delta / 2) / x_range_delta).astype(int)
        grid_y_indices = ((grid_y - y_range_delta / 2) / y_range_delta).astype(int)

        # sum = 0
        # true_coords = np.array(np.where(mask)).T
        # coords = {}
        # for (x0, y0), value in zip(true_coords, Z[grid_x_indices, grid_y_indices][mask]):
        #     print(f"({x_vals[x0]}, {y_vals[y0]}): {value}")
        #     sum += value

        # Calculate weighted area
        area = np.sum(Z[grid_x_indices, grid_y_indices][mask])

        vertices_to_area[flat_vertices] = area
        result.append(area)

    return np.array(result, dtype=float), vertices_to_area

# def calculate_weighted_area(vertices_col, Z, num_ticks_per_yard=3, vertices_to_area=None):
    
#     max_x = 120
#     max_y = 53.3
    
#     result = []

#     for vertices in vertices_col: 
    
#         vertices = sort_vertices_clockwise(vertices)
#         # print(f'vertices: {vertices}')
        
#         # if these vertices are cached, just return that result
#         if tuple(vertices.flatten()) in vertices_to_area.keys():
#             result.append(vertices_to_area[tuple(vertices.flatten())])
#             # print('return from cached')
#             continue

#         bounding_min_x, bounding_max_x = min(x for x, _ in vertices), max(x for x, _ in vertices)
#         bounding_min_y, bounding_max_y = min(y for _, y in vertices), max(y for _, y in vertices)

#         bounding_min_x = max(bounding_min_x - (1 / num_ticks_per_yard), 0)
#         bounding_max_x = min(bounding_max_x + (1 / num_ticks_per_yard), max_x)
#         bounding_min_y = max(bounding_min_y - (1 / num_ticks_per_yard), 0)
#         bounding_max_y = min(bounding_max_y + (1 / num_ticks_per_yard), max_y)
        
#         area = 0
#         for i in range(int(bounding_min_x * num_ticks_per_yard), int(bounding_max_x * num_ticks_per_yard)):
#                 for j in range(int(bounding_min_y * num_ticks_per_yard), int(bounding_max_y * num_ticks_per_yard)):
#                     x_val = round(0 + i * (1 / (num_ticks_per_yard)), 4)
#                     y_val = round(0 + j * (1 / (num_ticks_per_yard)), 4)
#                     x = round(x_val + (1 / (2 * num_ticks_per_yard)), 4)
#                     y = round(y_val + (1 / (2 * num_ticks_per_yard)), 4)
#                     if point_in_polygon(x, y, vertices):
#                         area += Z[i, j]

#         vertices_to_area[tuple(vertices.flatten())] = area
#         # print(f'calculated without cache', area)
#         result.append(area)

#     return np.array(result, dtype=float), vertices_to_area

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TPC Methods

def tackle_percentage_contribution_per_frame(frame_data:pd.DataFrame)->dict:
    """ 
    For every unique player attributed to a square on the defending team, take them out and see how much Voronoi area would be gained by the player in possession. 

    Params: 
    - frame_data (pd.DataFrame): a dataframe from the organize_game_data method with columns ['nflId', 'ballCarrierId', 'is_offense', 'x', 'y']

    Returns: 
    - dictionary with keys of nflId and value of the tackle percentage contribution for that frame
    """
    area_protected_blockers = {}
    area_protected_no_blockers = {}

    # get the ball carrier and offensive players
    ballCarrier = frame_data.ballCarrierId.iloc[0]
    x, y, dir, s = frame_data[frame_data.nflId==ballCarrier].iloc[0].loc[['x', 'y', 'dir', 's']]  # get the x, y, direction, and speed of the ball carrier for weighting method

    # calculate the weight of the field
    Z = calculate_Z_vectorized(x, y, dir, s)

    vertices_to_area = {}

    # calculate the weighted area of the ball carrier
    frame_data = calculate_voronoi_areas(frame_data)
    # unweighted_baseline = frame_data[frame_data.nflId==ballCarrier]['voronoi_area'].iloc[0]
    # print(f'unweighted baseline area no blockers: {unweighted_baseline}')

    frame_data['weighted_voronoi_area'] = float(0)
    frame_data.loc[frame_data.is_offense, 'weighted_voronoi_area'], vertices_to_area = calculate_weighted_area(vertices_col=frame_data[frame_data.is_offense]['vertices'].copy(), Z=Z, vertices_to_area=vertices_to_area)

    # frame_data['weighted_voronoi_area'] = frame_data.voronoi_area # (unweighted)
    baseline_area_no_blockers = frame_data.loc[frame_data.nflId==ballCarrier, 'weighted_voronoi_area'].iloc[0]
    
    frame_data = recognize_blockers(frame_data) # (toggle to recognize adjacent blockers or not)
    baseline_area_blockers = frame_data.loc[frame_data.nflId==ballCarrier, 'weighted_voronoi_area'].iloc[0] # baseline area of the ball carrier

    # print(f'weighted baseline area no blockers: {baseline_area_no_blockers}')

    ################ DEBUG ######################
    # print(baseline_area)
    # print(f'baseline area of ball carrier: {baseline_area}')
    # # print(f'initial blockers: {blockers}')
    # print(vertices_to_area)
     
    # iterate through the IDs of the players that are not offense
    for player_id in frame_data.loc[~frame_data['is_offense'], 'nflId'].unique():
        # print(f'\nremoving defender {player_id}...')

        # take the frame data if that player didn't exist
        filtered_frame_data = frame_data[frame_data.nflId != player_id].copy()
        # print(len(filtered_frame_data.nflId))

        # calculate the weighted voronoi areas of the ball carrier 
        filtered_frame_data = calculate_voronoi_areas(filtered_frame_data)
        # unweighted_updated = filtered_frame_data[filtered_frame_data.nflId==ballCarrier]['voronoi_area'].iloc[0]
        # print(f'updated unweighted area no blockers: {unweighted_updated}')

        filtered_frame_data['weighted_voronoi_area'] = float(0)
        filtered_frame_data.loc[filtered_frame_data.is_offense, 'weighted_voronoi_area'], vertices_to_area = calculate_weighted_area(vertices_col=filtered_frame_data[filtered_frame_data.is_offense]['vertices'].copy(), Z=Z, vertices_to_area=vertices_to_area)
        # frame_data['weighted_voronoi_area'] = frame_data.voronoi_area # (unweighted)
        protected_area_no_blockers = filtered_frame_data.loc[filtered_frame_data.nflId==ballCarrier, 'weighted_voronoi_area'].iloc[0]
        filtered_frame_data = recognize_blockers(filtered_frame_data) # (toggle to recognize adjacent blockers or not)
        protected_area_blockers = filtered_frame_data.loc[filtered_frame_data.nflId==ballCarrier, 'weighted_voronoi_area'].iloc[0] # baseline area of the ball carrier

        # print(f'updated weighted area no blockers: {protected_area_no_blockers}')

        # DEBUG
        # print(f'{player_id} removed, blockers: {blockers}, protected area: {protected_area}')
        # calculate how much additional space the offense gets
        # print(f'protected: {protected_area}, baseline: {baseline_area}')

        protected_area_blockers = max(protected_area_blockers, baseline_area_blockers) # handle floating point imprecision leading to small negatives
        protected_area_no_blockers = max(protected_area_no_blockers, baseline_area_no_blockers) # handle floating point imprecision leading to small negatives

        area_protected_no_blockers[player_id] = round(protected_area_no_blockers - baseline_area_no_blockers, 4)  # how much more area do they get, not factoring in blockers?
        area_protected_blockers[player_id] = round(protected_area_blockers - baseline_area_blockers, 4) # how much area do they get factoring in blockers?

    return area_protected_blockers, area_protected_no_blockers

def tackle_percentage_contribution_per_play(frame_dict:dict, filepath:str, animation:bool=False): 
    """
    This iterates through the frames in any given play and calculates the tackle percentage contribution of each player
    
    Params: 
    - frame_dict: dict from the organize_game_data method for each play
    - filepath (str): the filepath of the folder under which we are caching play data
    - animation (bool): whether or not we want to make an MP4 of the play

    Returns: 
    - dictionary with keys of nflId and value of the tackle percentage contribution for that play
    """
    # empty dict, one indexed by player, the other indexed by frame
    total_tpc = {}
    tpc_per_frame_blockers = {}
    tpc_per_frame_no_blockers = {}

    # sort the frames
    frame_dict_sorted = sorted(frame_dict.items(), key=lambda x: x[0])
    # iterate through the frames of the play
    for key, frame in frame_dict_sorted: 
        # print(f'\n**********************************************')
        # print(f'analyzing frame {key}...')
        # print(f'**********************************************\n')

        # get protected areas, append to both dictionaries
        frame_tpc_blockers, frame_tpc_no_blockers = tackle_percentage_contribution_per_frame(frame)
        tpc_per_frame_blockers[key] = frame_tpc_blockers
        tpc_per_frame_no_blockers[key] = frame_tpc_no_blockers

    #     # append to the overall dict for the play
    #     for player, contribution in frame_tpc_blockers.items():
    #         if player in total_tpc.keys(): 
    #             total_tpc[player] += contribution
    #         else: 
    #             total_tpc[player] = contribution
    
    # # normalize every player's contribution such that it sums to 1
    # total_protected_area = sum(total_tpc.values())
    # for key, value in total_tpc.items():
    #     total_tpc[key] = value / total_protected_area


    # Convert the dictionary with the frame data to a DataFrame to cache
    # The keys of the outer dict become the index, and the inner dicts' keys become the column names
    tpc_per_frame_blockers_df = pd.DataFrame.from_dict(tpc_per_frame_blockers, orient='index')
    tpc_per_frame_no_blockers_df = pd.DataFrame.from_dict(tpc_per_frame_no_blockers, orient='index')
    
    # rowsums to 1
    row_sums_blockers = tpc_per_frame_blockers_df.sum(axis=1)
    tpc_per_frame_blockers_df = tpc_per_frame_blockers_df.div(row_sums_blockers, axis=0)

    row_sums_no_blockers = tpc_per_frame_no_blockers_df.sum(axis=1)
    tpc_per_frame_no_blockers_df = tpc_per_frame_no_blockers_df.div(row_sums_no_blockers, axis=0)

    # Save to CSV, with the index to make future multiplication easier
    # tpc_per_frame_df.to_csv(f'{filepath}/tpc_per_frame_weighted.csv', index=True)
    tpc_per_frame_blockers_df.to_csv(f'{filepath}/tpc_per_frame_weighted_blockers.csv', index=True)
    tpc_per_frame_no_blockers_df.to_csv(f'{filepath}/tpc_per_frame_weighted_no_blockers.csv', index=True)

    # cast everything to strings from int64 (otherwise cannot store in JSON)
    # total_tpc_converted = {int(key): value for key, value in total_tpc.items()}

    # cache this result as a JSON for each play
    # json.dump(total_tpc_converted, open(filepath+'/tpc_weighted.json', 'w'))
    # json.dump(total_tpc_converted, open(filepath+'/tpc_weighted_blockers.json', 'w'))
    
    # if the animation method is called
    if animation: 

        # open plots were taking too much memory
        plt.close('all')
        
        def animate(frame_number:int, ax:plt.Axes):
            """
            frame_number (int): the number of the frame, pulling from the nonlocal frame_dict 
            ax (plt.ax): passing the axis for the whole gif into the method, doesn't work otherwise (not sure why)
            """
            # Get the dataframe for the current frame
            current_frame = frame_dict[frame_number]

            # Call the calculate_voronoi_areas function with plot_graph=True
            calculate_voronoi_areas(current_frame, plot_graph=True, tpc_dict=tpc_per_frame_blockers[frame_number], ax=ax)

        fig, ax = plt.subplots(figsize=(24, 16))
        ani = FuncAnimation(fig, lambda x: animate(x, ax), frames=sorted(frame_dict.keys()), repeat=False)
        
        # Save the animation
        ani.save(filepath + '/voronoi_visualizer_weighted_blockers.mp4', writer='ffmpeg')

    return {}

def analyze_play(key:int, play:dict, filepath:str, animation:bool):
    """
    Wrapper function to analyze a single play. This function will be executed in parallel.
    Params: 
    - key (int): the play number
    - play (dict): a dictionary of the frames (pd.DataFrame) of each play
    - filepath (str): the filepath of tha game within which we will save information/animations of the play
    """
    print(f'analyzing play {key}...')  # For debugging purposes

    # Define the play's file path
    play_filepath = f'{filepath}/{key}'
    if not os.path.exists(play_filepath):
        os.makedirs(play_filepath)

    try:
        # Calculate the tackle_percentage_contribution
        # Ensure that the tackle_percentage_contribution_per_play function is defined appropriately
        play_tpc = tackle_percentage_contribution_per_play(frame_dict=play, filepath=play_filepath, animation=animation)

        return {}
    except Exception as e:
        print(f'Error processing play {key}: {e}')
        return {}

def analyze_game(game_id, tracking_file, plays_file='./data/plays.csv', game_file='./data/games.csv', animation:bool=False):
    """ 
    A method to analyze a game. Calling this will analyze and cache all the plays + the results of the analysis
    Param: 
    - game_id (int): the ID of the game as found in the Kaggle cleaned data
    - tracking_file (str): the address of the file in which the tracking data is stored
    - plays_file (str): the address of the plays file
    - game_file (str): the filepath of the file containing information about each game
    """
    print(f'analyzing game {game_id}...')
    
    games = pd.read_csv(game_file)
    game_data = games[games.gameId==game_id].iloc[0, [0, 5, 6]] # pull the ID (col 0), home team (col 5), visitng team (col 6)
    filepath = f'./games/{game_data.iloc[0]}_{game_data.iloc[1]}_{game_data.iloc[2]}'

    # Create a directory for the game if none exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Sort and organize the data (eval df)
    valid_plays = pd.read_csv('./data/eval_frame_df.csv')
    valid_plays = valid_plays[valid_plays.gameId==game_id]
    if valid_plays.empty: 
        return {}
    game_data_organized = organize_game_data_eval_df(load_game_data(tracking_file, plays_file, game_id), valid_plays)
    sorted_game_data_organized = sorted(game_data_organized.items(), key=lambda x: x[0])
        
    # # Sort and organize the data (no eval df)
    # game_data_organized = organize_game_data(load_game_data(tracking_file, plays_file, game_id))
    # sorted_game_data_organized = sorted(game_data_organized.items(), key=lambda x: x[0])


    # Dictionary to store the overall tackle_percentage_contribution
    # game_tpc = {}

    # Using ProcessPoolExecutor to parallelize the loop
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_play, key, play, filepath, animation) for key, play in sorted_game_data_organized]

        for future in as_completed(futures):
            play_tpc = future.result()

    # Convert game_tpc keys from int64 to string to store in JSON
    # game_tpc_converted = {int(key): value for key, value in game_tpc.items()}

    # Cache this result as a JSON for each game
    # json.dump(game_tpc_converted, open(filepath + '/game_tpc.json', 'w'))
    # json.dump(game_tpc_converted, open(filepath + '/game_tpc_weighted_blockers.json', 'w'))

    return {}

def analyze_game_unparallelized(game_id, tracking_file, plays_file='./data/plays.csv', game_file='./data/games.csv', animation:bool=False):
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
    valid_plays = pd.read_csv('./data/eval_frame_df.csv')
    valid_plays = valid_plays[valid_plays['gameId']==game_id]
    print(valid_plays)
    game_data_organized = organize_game_data(load_game_data(tracking_file, plays_file, game_id), valid_plays)
    sorted_game_data_organized = sorted(game_data_organized.items(), key=lambda x: x[0])

    # Dictionary to store the overall tackle_percentage_contribution
    game_tpc = {}

    for key, play in sorted_game_data_organized: 
        play_tpc = analyze_play(key, play, filepath, animation)
        for player, contribution in play_tpc.items():
            game_tpc[player] = game_tpc.get(player, 0) + contribution

    # Convert game_tpc keys from int64 to string to store in JSON
    game_tpc_converted = {str(key): value for key, value in game_tpc.items()}

    # Cache this result as a JSON for each game
    json.dump(game_tpc_converted, open(filepath + '/game_tpc_voronoi_unweighted.json', 'w'))

    return game_tpc

def analyze_play_unparallelized(play_number, game_id, tracking_file, plays_file='./data/plays.csv', players_file='./data/players.csv', game_file='./data/games.csv', animation:bool=False):
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

    for key, play in sorted_game_data_organized: 
        if key != play_number: 
            continue
        play_tpc = analyze_play(key, play, filepath, animation)
        for player, contribution in play_tpc.items():
            game_tpc[player] = game_tpc.get(player, 0) + contribution

    # Convert game_tpc keys from int64 to string to store in JSON
    game_tpc_converted = {str(key): value for key, value in game_tpc.items()}

    # Cache this result as a JSON for each game
    json.dump(game_tpc_converted, open(filepath + '/game_tpc.json', 'w'))

    return game_tpc

def build_cpr_no_nan(a:bool, b:bool, c:bool, d:bool):
    """
    Build the constrictive presence ratio CSV
    a: weighted_blockers
    b: weighted_no_blockers
    c: unweighted_blockers
    d: unweighted_no_blockers
    """    
    # Read the 'eval_frame_df.csv' file
    eval_frame_df = pd.read_csv(f'./data/eval_frame_df.csv')
    finished = set()

    # Iterate through rows of 'eval_frame_df'
    for _, row in eval_frame_df.iterrows():
        gameId = int(row['gameId'])
        playId = int(row['playId'])
        if (gameId, playId) in finished: 
            continue
        
        # Look up the game in 'games.csv'
        games_df = pd.read_csv('./data/games.csv')
        game_info = games_df[(games_df['gameId'] == gameId)]
        print(game_info)

        if game_info.empty:
            print(f"No game found for gameId: {gameId}")
            continue

        homeTeamAbbr = game_info['homeTeamAbbr'].values[0]
        visitingTeamAbbr = game_info['visitorTeamAbbr'].values[0]

        # Define the folder path
        folder_path = f'./games/{gameId}_{homeTeamAbbr}_{visitingTeamAbbr}/{playId}/'

        # Check if the folder exists and if the file exists
        if not os.path.exists(folder_path):
            print(f"Folder not found for gameId: {folder_path}")
            finished.add((gameId, playId))
            continue

        if a: 
            weighted_blockers = os.path.join(folder_path, 'tpc_per_frame_weighted_blockers.csv')
        if b:
            weighted_no_blockers = os.path.join(folder_path, 'tpc_per_frame_weighted_no_blockers.csv')
        if c: 
            unweighted_no_blockers = os.path.join(folder_path, 'tpc_per_frame_unweighted_no_blockers.csv')
        if d: 
            unweighted_blockers = os.path.join(folder_path, 'tpc_per_frame_unweighted.csv')

        if not (os.path.exists(weighted_blockers) and os.path.exists(weighted_no_blockers)):
            print(f"File not found for gameId: {gameId}, {playId}")
            print(weighted_blockers)
            finished.add((gameId, playId))
            continue

        # Read 'tpc_per_frame_weighted_blockers.csv'
        if a: 
            tpc_per_frame_weighted_blockers = pd.read_csv(weighted_blockers, index_col=0)
            # tpc_per_frame_weighted_blockers = tpc_per_frame_weighted_blockers.fillna(0)
            if tpc_per_frame_weighted_blockers.isna().any().any():
                continue
        if b: 
            tpc_per_frame_weighted_no_blockers = pd.read_csv(weighted_no_blockers, index_col=0)
            # tpc_per_frame_weighted_no_blockers = tpc_per_frame_weighted_no_blockers.fillna(0)
            if tpc_per_frame_weighted_no_blockers.isna().any().any():
                continue
        if c: 
            tpc_per_frame_unweighted_blockers = pd.read_csv(unweighted_blockers, index_col=0)
            tpc_per_frame_unweighted_blockers = tpc_per_frame_unweighted_blockers.fillna(0)
        if d: 
            tpc_per_frame_unweighted_no_blockers = pd.read_csv(unweighted_no_blockers, index_col=0)
            tpc_per_frame_unweighted_no_blockers = tpc_per_frame_unweighted_no_blockers.fillna(0)

        # Sort 'eval_frame_df' by matching gameId and playId and sort it in order of frameId
        sorted_eval_frame_df = eval_frame_df[(eval_frame_df['gameId'] == gameId) & (eval_frame_df['playId'] == playId)].sort_values(by='frameId')
        filtered_eval_frame_df = sorted_eval_frame_df[sorted_eval_frame_df['frameId'].isin(tpc_per_frame_weighted_no_blockers.index)]
        ######### DEBUG #########################
        # print(filtered_eval_frame_df.index)

        # Calculate the product
        expected_yards_diff = filtered_eval_frame_df['expectedYardsByCarrier'].diff(-1).fillna(0)
        expected_yards_diff.index = tpc_per_frame_weighted_blockers.index

        # Check if the length of expected_yards_diff is the same as the number of rows in tpc_per_frame_weighted_blockers
        if len(expected_yards_diff) == len(tpc_per_frame_weighted_blockers):
            # Multiply each entry in the corresponding row by the value in expected_yards_diff
            if a: 
                cpr_weighted_blockers = tpc_per_frame_weighted_blockers.multiply(expected_yards_diff, axis=0)
            if b: 
                cpr_weighted_no_blockers = tpc_per_frame_weighted_no_blockers.multiply(expected_yards_diff, axis=0)
            if c: 
                cpr_unweighted_blockers = tpc_per_frame_unweighted_blockers.multiply(expected_yards_diff, axis=0)
            if d: 
                cpr_unweighted_no_blockers = tpc_per_frame_unweighted_no_blockers.multiply(expected_yards_diff, axis=0)
        else:
            # Print the shapes of both dataframes
            print(f"Shapes do not match - expected_yards_diff shape: {len(expected_yards_diff)}, tpc_per_frame_weighted_blockers shape: {len(tpc_per_frame_weighted_blockers)}")
            finished.add((gameId, playId))

        if a:
            cpr_weighted_blockers.to_csv(os.path.join(folder_path, 'cpr_weighted_blockers.csv'))
        if b: 
            cpr_weighted_no_blockers.to_csv(os.path.join(folder_path, 'cpr_weighted_no_blockers.csv'))
        if c: 
            cpr_unweighted_blockers.to_csv(os.path.join(folder_path, 'cpr_unweighted_blockers.csv'))
        if d: 
            cpr_unweighted_no_blockers.to_csv(os.path.join(folder_path, 'cpr_unweighted_no_blockers.csv'))

        finished.add((gameId, playId))

        print(f'success for {gameId}, {playId}')
