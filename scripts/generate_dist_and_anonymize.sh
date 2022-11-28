DATA=$1
NSG=$2
WORKERS=$3
LOG=$4


python generate_pairwise_distances.py --data=$DATA --strategy=mean --n_sg=$NSG --workers=$WORKERS --log=$LOG


python run_anonymization.py --data=$DATA --strategy=mean --n_sg=$NSG --calgo=km --galgo=ad2 --enforcer=soa --k_list=2,4,6,8,10 --w_list=-1 --l_list=1 --max_dist_list=1 --anony_mode=only_clusters --log=$LOG --log_modes=con --reset_w_list=-1 --workers=$WORKERS

python run_anonymization.py --data=$DATA --strategy=mean --n_sg=$NSG --calgo=km --galgo=ad2 --enforcer=soa --k_list=10 --w_list=-1 --l_list=2,3,4 --max_dist_list=1 --anony_mode=only_clusters --log=$LOG --log_modes=con --reset_w_list=-1 --workers=$WORKERS

python run_anonymization.py --data=$DATA --strategy=mean --n_sg=$NSG --calgo=km --galgo=ad2 --enforcer=soa --k_list=10 --w_list=-1 --l_list=4 --max_dist_list=0,0.25,0.5,0.75 --anony_mode=only_clusters --log=$LOG --log_modes=con --reset_w_list=-1 --workers=$WORKERS

python run_anonymization.py --data=$DATA --strategy=mean --n_sg=$NSG --calgo=km --galgo=ad2 --enforcer=soa --k_list=10 --w_list=-1 --l_list=4 --max_dist_list=1 --anony_mode=only_clusters --log=$LOG --log_modes=con --reset_w_list=4,9,5,10 --workers=$WORKERS