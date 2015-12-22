alg=$1
move_cost=$2
turn_cost=$3

python run.py ../data/tinySearch.txt eatdot1 $move_cost $turn_cost
python run.py ../data/smallSearch.txt eatdot1 $move_cost $turn_cost
python run.py ../data/mediumSearch.txt eatdot1 $move_cost $turn_cost

