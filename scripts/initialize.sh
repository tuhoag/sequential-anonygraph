DATA=$1
NSG=$2
WORKERS=$3
LOG=$4

python initialize_raw_subgraphs.py --data=$DATA --strategy=mean --n_sg=$NSG --workers=$WORKERS --log=$LOG
python generate_pairwise_distances.py --data=$DATA --strategy=mean --n_sg=$NSG --workers=$WORKERS --log=$LOG