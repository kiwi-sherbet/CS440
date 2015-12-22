alg=$1
move_cost=$2
turn_cost=$3

python run.py ../data/smallTurns.txt $alg $move_cost $turn_cost
python run.py ../data/mediumMaze.txt $alg $move_cost $turn_cost
python run.py ../data/bigMaze.txt $alg $move_cost $turn_cost
python run.py ../data/openMaze.txt $alg $move_cost $turn_cost
