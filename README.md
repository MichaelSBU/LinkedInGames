Prerequisite libaries:
matplotlib, pandas, numpy, seaborn

The project pertians to 5 LinkedIn games

1. Zip:

The following is relevan to the Zip directory!

To run a Zip game add a game to the games directory in the specified format:

form the grid using commas.
,,,,,
,,,,,
,,,,,
,,,,,
,,,,,
,,,,,
and fill in the positions of the numbered cells, using there number.
,,,,,
,,3,4,,
,8,,,5,
,1,,,6,
2,,,,,7
,,,,,

To solve a game, use the following command:

python3 Zip.py games/zip{i}.txt {h}

Where i is particular zip game you want to run, and h dictates which heuristic you might want to use.
For a standard DFS, you can omit h. For A* MST heuristic, use 2. For A* Enhanced heuristic, use 3.

Then, an animation should appear on your screen of the algorithm solving the game board specified by the input file.

2. Queens:

The following is relevan to the Queens directory!

To run a Queens game add a game to the games directory in the specified format:

form the grid using commas.
,,,,,,,
,,,,,,,
,,,,,,,
,,,,,,,
,,,,,,,
,,,,,,,
,,,,,,,
,,,,,,,
and fill in the colored region, using arbitrary, but consistent, numbers of 0 to n-1 (each mapping to 1 of the n colors).
3,3,3,7,7,0,1,1
3,3,4,7,0,0,0,1
3,3,4,4,0,0,0,1
3,3,6,4,5,0,1,1
3,6,6,6,5,5,1,1
3,3,6,2,2,1,1,1
3,3,3,2,2,2,1,1
3,3,3,2,2,1,1,1


To solve a game, use the following command:

python3 Queens.py games/queens{i}.txt {a}

Where i is particular queens game you want to run, and a dictates the algorithm you want to use.
For a standard DFS, use 0. For row-col based variable formualtion with FC and MCV use 1. For for color based variable formulation with FC and MCV use 2.

Then, an animation should appear on your screen of the algorithm solving the game board specified by the input file.

3. Tango:

The following is relevan to the Tango directory!

To run a Queens game add a game to the games directory in the specified format:

form the grid using commas.
,,,,,,,
,,,,,,,
,,,,,,,
,,,,,,,
,,,,,,,
,,,,,,,
,,,,,,,
,,,,,,,
and fill in the symbol placement, using arbitrary, but consistent, numbers 1 or 2 (each mapping to either sun or moon symbols). To add explicit constraints ('x' or '=' of adjacent cells) form them using x1,y1{x|=}x2,y2 where x1 and x2 are 0-indexed row coordinates and y1 and y2 are 0-indexed column coordinates such that cell (x1,y1) is adjacent (non-diagonally) to cell (x2,y2), where 'x' indicates those adjacent cells must be different symbols and '=' indicates those cells must be the same symbols.
2,,,2,,
,,1,,,
,,,,,2
1,,,,,2
,,,1,,
2,,,,1,

1,0×1,1
1,3=1,4
1,4×2,4
3,0=4,0


To solve a game, use the following command:

python3 Tango.py games/tango{i}.txt {a}

Where i is particular queens game you want to run, and a dictates the algorithm you want to use.
For a standard DFS, use 0. For FC and MCV use 1.

Then, an animation should appear on your screen of the algorithm solving the game board specified by the input file.

4. Pinpoint
Experiment performed manually

5. Crossclimb
Experiment performed manually

Plotting

In all directories, a plotting.ipynb file should exist, containing the data from our expeirments and code to visualize our results.