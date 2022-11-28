DATA=yago
NSG=50
max_dist_LIST=$1
K_LIST=$2
W_LIST=$3
WORKERS=$4

python run_generate_sequence_subgraphs_and_pair_dist.py --data=$DATA --sample=-1 --strategy=equaladd --n_sg=$NSG --info_loss=adm --alpha_adm=0.5 --alpha_dm=0.5 --workers=$WORKERS --log=i

python run_anonymization.py  --data=$DATA --sample=-1 --strategy=equaladd --n_sg=$NSG --info_loss=adm --alpha_adm=0.5 --alpha_dm=0.5 --calgo=km --max_dist_list=$max_dist_LIST --k_list=$K_LIST --w_list=$W_LIST --galgo=ad --log=i --workers=$WORKERS

python run_generate_sequence_subgraphs_and_pair_dist.py --data=$DATA --sample=-1 --strategy=equalraw --n_sg=$NSG --info_loss=adm --alpha_adm=0.5 --alpha_dm=0.5 --workers=$WORKERS --log=i

python run_anonymization.py  --data=$DATA --sample=-1 --strategy=equalraw --n_sg=$NSG --info_loss=adm --alpha_adm=0.5 --alpha_dm=0.5 --calgo=km --max_dist_list=$max_dist_LIST --k_list=$K_LIST --w_list=$W_LIST --galgo=ad --log=i --workers=$WORKERS