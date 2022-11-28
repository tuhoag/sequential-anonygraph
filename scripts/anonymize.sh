#!/bin/bash

function anonymize() {

    DATA=$1
    NSG=$2

    CALGO=$3
    GALGO=$4
    ENFORCER=$5
    K_LIST=$6
    L_LIST=$7
    MAX_DIST_LIST=$8
    RESET_W_LIST=$9
    MODE=$10
    WORKERS=$11
    LOG=$12

    echo $DATA
    echo $NSG
    echo $CALGO
    echo $GALGO
    echo $ENFORCER
    echo $K_LIST
    echo $L_LIST
    echo $MAX_DIST_LIST
    echo $RESET_W_LIST
    echo $MODE
    # echo $LOG

    # python run_anonymization.py --data=$DATA --strategy=mean --n_sg=$NSG --calgo=$CALGO --galgo=$GALGO --enforcer=$ENFORCER --k_list=$K_LIST --w_list=-1 --l_list=$L_LIST --max_dist_list=$MAX_DIST_LIST --anony_mode=$MODE --log=$LOG --log_modes=con --reset_w_list=$RESET_W_LIST --workers=$WORKERS
}

function anonymizeAllSettings() {
    DATA=$1
    WORKERS=$2

    anonymize $DATA 50 km ad2 soa 2,4,6,8,10 1,2,3,4 0,0.25,0.5,0.75,1 -1 only_clusters $WORKERS i
}

function anonymizeNormalCase() {
    DATA=$1
    WORKERS=$2
    LOG=$3

    # echo $DATA
    # echo $WORKERS
    # echo $LOG

    anonymize $DATA 50 km ad2 soa 2,4,6,8,10 1 1 -1 only_clusters -1 $WORKERS $LOG

    # anonymize $DATA 50 km ad2 soa 10 1,2,3,4,5 1 only_clusters -1 $WORKERS $LOG

    # anonymize $DATA 50 km ad2 soa 10 5 0,0.25,0.5,0.75,1 only_clusters -1 $WORKERS $LOG

    # anonymize $DATA 50 km ad2 soa 10 5 1 only_clusters 5,10 $WORKERS $LOG
}

function anonymizeTestCase() {
    DATA=$1
    WORKERS=$2

    anonymize $DATA 2 km ad2 soa 2,4,6,8,10 1,2,3,4 0,0.25,0.5,0.75,1 -1 only_clusters $WORKERS i
}


COMMAND=$1

if [ $COMMAND = "normal" ]; then
    DATA=$2
    WORKERS=$3
    LOG=$4

    anonymizeNormalCase $DATA $WORKERS $LOG
elif [ $COMMAND = "all" ]; then
    DATA=$2
    WORKERS=$3

    anonymizeAllSettings $DATA $WORKERS

elif [ $COMMAND = "test" ]; then
    DATA=$2
    WORKERS=$3

    anonymizeTestCase $DATA $WORKERS
fi