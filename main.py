import argparse
import logging

# from .generate_raw_dynamic_graph import add_arguments
import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
from anonygraph.constants import *

logger = logging.getLogger(__file__)

def setup_arguments():
    parser = argparse.ArgumentParser()
    rutils.add_log_argument(parser)

    parser.add_argument("command")
    parser.add_argument("subcommand")
    parser.add_argument("--data")
    parser.add_argument("--n_sg")
    parser.add_argument("--enforcer")
    parser.add_argument("--workers")

    args, _ = parser.parse_known_args()

    logger.info(args)
    params = {}
    for arg in vars(args):
        params[arg] = getattr(args, arg)

    return params

def anonymize_normal_case(args):
    logger.debug(args)

    # python run_anonymization.py --data=$DATA --strategy=mean --n_sg=$NSG --calgo=$CALGO --galgo=$GALGO --enforcer=$ENFORCER --k_list=$K_LIST --w_list=-1 --l_list=$L_LIST --max_dist_list=$MAX_DIST_LIST --anony_mode=$MODE --log=$LOG --log_modes=con --reset_w_list=$RESET_W_LIST --workers=$WORKERS

    common_args = args.copy()
    common_args["strategy"] = MEAN_EDGES_STRATEGY
    common_args["calgo"] = KMEDOIDS_CLUSTERING
    common_args["galgo"] = ADD_REMOVE_EDGES2_GEN
    # common_args["enforcer"] = SPLIT_OVERLAP_ASSIGNMENT
    common_args["w_list"] = "-1"
    common_args["anony_mode"] = "only_clusters"

    new_args = common_args.copy()
    new_args["k_list"] = [2,4,6,8,10]
    new_args["l_list"] = "1"
    new_args["max_dist_list"] = "1"
    new_args["reset_w_list"] = "-1"
    rutils.run_anonymization_runner(new_args)

    new_args = common_args.copy()
    new_args["k_list"] = [10]
    new_args["l_list"] = [1,2,3,4,5]
    new_args["max_dist_list"] = [1]
    new_args["reset_w_list"] = [-1]
    rutils.run_anonymization_runner(new_args)

    new_args = common_args.copy()
    new_args["k_list"] = [10]
    new_args["l_list"] = [5]
    new_args["max_dist_list"] = [0, 0.25, 0.5, 0.75, 1]
    new_args["reset_w_list"] = [-1]
    rutils.run_anonymization_runner(new_args)

    new_args = common_args.copy()
    new_args["k_list"] = [10]
    new_args["l_list"] = [5]
    new_args["max_dist_list"] = [1]
    new_args["reset_w_list"] = [1,2,5,10]
    rutils.run_anonymization_runner(new_args)

def main(args):
    logger.info(args)
    command = args["command"]
    subcommand = args["subcommand"]

    if command == "anonymize":
        if subcommand == "normal":
            anonymize_normal_case(args)
        else:
            raise Exception("Unsupported command '{}' '{}'".format(command, subcommand))
    else:
        raise Exception("Unsupported command '{}'".format(command))

if __name__ == "__main__":
    logger.info("test")
    args = setup_arguments()
    # args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)