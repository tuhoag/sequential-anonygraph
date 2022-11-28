DATA=email-temp
NSG=50
max_dist_LIST=$1
K_LIST=$2
W_LIST=$3
WORKERS=$4

python run_anonymization.py --data=$DATA --strategy=mean --n_sg=$NSG --sattr=dept --calgo=km --galgo=ad2 --enforcer=msa --k_list=2,4,6,8,10 --w_list=-1 --l_list=1 --max_dist_list=1 --anony_mode=only_clusters --log=i --log_modes=con --reset_w_list=-1 --workers=15

python run_anonymization.py --data=$DATA --strategy=mean --n_sg=$NSG --sattr=dept --calgo=km --galgo=ad2 --enforcer=msa --k_list=10 --w_list=-1 --l_list=2,3,4 --max_dist_list=1 --anony_mode=only_clusters --log=i --log_modes=con --reset_w_list=-1 --workers=15

python run_anonymization.py --data=$DATA --strategy=mean --n_sg=$NSG --sattr=dept --calgo=km --galgo=ad2 --enforcer=msa --k_list=10 --w_list=-1 --l_list=4 --max_dist_list=0,0.25,0.5,0.75 --anony_mode=only_clusters --log=i --log_modes=con --reset_w_list=-1 --workers=15

python run_anonymization.py --data=$DATA --strategy=mean --n_sg=$NSG --sattr=dept --calgo=km --galgo=ad2 --enforcer=msa --k_list=10 --w_list=-1 --l_list=4 --max_dist_list=1 --anony_mode=only_clusters --log=i --log_modes=con --reset_w_list=4,9,5,10 --workers=15