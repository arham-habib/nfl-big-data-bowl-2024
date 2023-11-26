# nfl-big-data-bowl-2024

'nfl_big_data_bowl.ipynb' is where the majority of code is
the './games/' folder contains every game's data. It contains sub folders for each play

- [X] Change the visualization such that instead of centroid, use the position of the player
    - add a dot where the player is
    - label the player in possession at the top of the graph
    - label the players offense or defense
- [ ] Address that grouping the offensive player makes defensive calculations wackm
    - make it such that if they're touching the space of another offensive player they become one unit?
    - something is wrong in play 167 with player 41239
- [ ] Create a labeled_tpc.json that has the names of each player instead of the ID in the analyze_game method
