# nfl-big-data-bowl-2024

## Updates: 
- Data analysis: 

    - 'nfl_big_data_bowl.ipynb' is where the majority of the ideation is
    - 'voronoi_analysis.py' runs a parallelized voronoi tesselation for each play, takes ~1hr to run for 135 games
    - 'distance_analysis.py' runs a parallelized algorithm to get the euclidean distance of each defending player from the ball_carrier

    - frame by frame defended_area breakdowns for each player are in 'tpc_per_frame_updated_voronoi.csv'
    - frame by frame euclidean distances are in 'distances_per_frame.csv'
    - the overall TPCs for this play (summed over frames and divided by total, so converted to a percent) is in 'tpc_updated_voronoi.json'
    - 'distance_weighted_tpc.csv/json' are the dot products of the distance_per_frame and tpc_per_frame
    - voronoi_visualizer.mp4 is a gif of the play (I didn't do this for everything since it'd take forever and we can't review them all anyway)
    - rest is old stuff

- Utils: 
    - latest version in 'utils.py' 
    - had to use a python file instead of a notebook to parallelize this
    - segmented 

## Things to discuss

- [ ] I think the way we're filtering plays by eliminating running plays is leading to sample bias
    - I think we should also include these plays after the ball is handed off and see if the data makes more sense

- [ ] Change the weighting of TPC based on the "relative value" of the space defended

    - 1. linearly decrease the value based on the Euclidean distance between the ball carrier and the defender
        - takes the form TPC_adj_{player_id} = TPC_{player} + B_1 * euclidean_distance_{player}, where B_1 should be negative
        - straightforward to calculate regress tackles + assits to every player's TPC and distace values of each game, only question is how many "tackles" are assists worth?
        - Alternatively, instead of doing this play by play, we can regress it game to game
        - data highly available (back of envelope, 135 games * 60 plays/game * 11 players/play = 89,100 data points to calibrate B_1)
        - might not capture the nuances of differnet Euclidean distances being different values
        - might not capture the nonlinearity of how value of defended territory decreases (decay is probably exponential for the first 10 ft, then levels off)

    - 2. map Euclidean distance to an exponential decay function
        - form: e^-(distance_{player}/constant) * TPC_{player} = TPC_adj_{player}
        - Captures the nonlinearity of how the value of defended territory decreases
        - Might not capture the differences of different Euclidean distances being different (defending 10 away in X-axis vs Y-axis)
        - We have to estimate two parameters, one of which is nonlinear, so standard regression goes out the window (IDK how to solve this closed form tbh)

    - 3. break the distance from ball_carrier into an x component and y component, and separately calculate a loss penalty for both
        - takes the form TPC_adj_{player_id} = TPC_{player} + B_1 * x_distance_{player} + y_distance_{player}, where B_1 and B_2 should be negative and |B_2| < |B_1| 
        - captures that different axes are more or less relevant
        - still runs into problems with linear decay (again, the process is probably exponential at first)
        - since we're estimating two parameters and not 1, we probably have to use play by play data, not game to game

    - 4. Create a function f(x,y) = worth(point) and just integrate over x and y for the area
        - would probably create a map from cartesian coordinates to adj_value coordinates to do this most efficiently
        - so need insanely good priors, unsure how to implement

    - 5. add some interaction terms e.g. tackles ~ euclidean_distance, euclidean_distance * TPC

    - Rough ranking of approaches IMO: 3, 2, 1, 5, 4

- [ ] Address that grouping the offensive player makes defensive calculations wack
    - make it such that if they're touching the space of another offensive player they become one unit?
    - has become easier now that I'm not breaking it into discrete blocks tbh, that was a stupid idea on my part

## TODO: 
 - change the voronoi animation to make the defenders and offensive players the same 
 - add acceleration and velocity vectors of the ball_carrier
