from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import logging
import os
import itertools

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.time_graph_generators as generators
import anonygraph.algorithms.clustering as calgo

logging.getLogger('sklearn_extra').setLevel(logging.WARNING)
logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_graph_generalization_argument(parser)
    rutils.add_log_argument(parser)

    parser.add_argument('--k_range', type=rutils.string2range(int))
    parser.add_argument('--k_list', type=rutils.string2list(int))
    parser.add_argument('--w_range', type=rutils.string2range(int))
    parser.add_argument('--w_list', type=rutils.string2list(int))
    parser.add_argument('--max_dist_list', type=rutils.string2list(float))
    # parser.add_argument('--max_dist_range', type=rutils.string2list(float))
    parser.add_argument('--l_range', type=rutils.string2range(int))
    parser.add_argument('--l_list', type=rutils.string2list(int))
    parser.add_argument("--calgo_list", type=rutils.string2list(str))
    parser.add_argument("--reset_w_list", type=rutils.string2list(int))

    parser.add_argument('--workers', type=int)
    parser.add_argument("--run_mode")

def get_args_values(args_name, args):
    if args_name in ["k", "l"]:
        list_args_name = "{}_list".format(args_name)
        range_args_name = "{}_range".format(args_name)

        if args[list_args_name] is not None and args[range_args_name] is not None:
            raise Exception("{} and {} cannot be passed at the same time.".format(list_args_name, range_args_name))
        elif args[list_args_name] is None and args[range_args_name] is None:
            logger.warning("{} and {} are not passed. Default values will be used.".format(list_args_name, range_args_name))
            args_values = None
        else:
            if args[list_args_name] is not None:
                args_values = args[list_args_name]
            else:
                args_values = args[range_args_name]
    elif args_name in ["calgo", "reset_w", "max_dist"]:
        list_args_name = "{}_list".format(args_name)
        args_values = args[list_args_name]
        logger.debug(args[list_args_name])
        logger.debug(args_values)

    if args_values is None:
        args_values = get_default_args_values(args_name)

    return args_values

def get_default_args_values(args_name):
    if args_name == "k":
        return [2,4,6,8,10]
    elif args_name == "l":
        return [1,2,3,4]
    elif args_name == "max_dist":
        return [0,0.25,0.5,0.75,1]
    elif args_name == "calgo":
        return ["km", "hdbscan"]
    elif args_name == "reset_w":
        return [-1, 1, 2, 3, 4]
    else:
        raise Exception("There is no default values for args: {}".format(args_name))

def check_args_existed(new_args, args_list):
    for current_args in args_list:
        if new_args == current_args:
            return True

    return False

def generate_normal_run_mode_args_list(args):
    k_values = get_args_values("k", args)
    logger.debug("k_values: {}".format(k_values))

    l_values = get_args_values("l", args)
    logger.debug("l_values: {}".format(l_values))

    threshold_values = get_args_values("max_dist", args)
    logger.debug("threshold_values: {}".format(threshold_values))

    calgo_names = get_args_values("calgo", args)
    logger.debug("calgo_list: {}".format(calgo_names))

    reset_w_values = get_args_values("reset_w", args)
    logger.debug("reset_w values: {}".format(reset_w_values))

    min_l = min(l_values)
    max_threshold = max(threshold_values)
    w = -1
    min_reset_w = min(reset_w_values)

    args_list = []
    existed_args = set()
    # evaluate k
    for k, calgo_name in itertools.product(k_values, calgo_names):
        current_args = args.copy()

        new_args = {
            "k": str(k),
            "w": str(w),
            "max_dist": str(max_threshold),
            "l": str(min_l),
            "reset_w": str(min_reset_w),
            "calgo": calgo_name,
        }
        current_args.update(new_args)

        if not check_args_existed(current_args, args_list):
            args_list.append(current_args)

    logger.debug("There are {} args_list for evaluating k".format(len(args_list)))

    # evaluate l
    max_k = max(k_values)
    for l, calgo_name in itertools.product(l_values, calgo_names):
        current_args = args.copy()

        new_args = {
            "k": str(max_k),
            "w": str(w),
            "max_dist": str(max_threshold),
            "l": str(l),
            "reset_w": str(min_reset_w),
            "calgo": calgo_name,
        }

        current_args.update(new_args)
        if not check_args_existed(current_args, args_list):
            args_list.append(current_args)

    logger.debug("There are {} args_list for evaluating k, l".format(len(args_list)))
    # evaluate max_dist
    max_l = max(l_values)
    for threshold, calgo_name in itertools.product(threshold_values, calgo_names):
        current_args = args.copy()

        new_args = {
            "k": str(max_k),
            "w": str(w),
            "max_dist": str(threshold),
            "l": str(max_l),
            "reset_w": str(min_reset_w),
            "calgo": calgo_name,
        }

        current_args.update(new_args)
        if not check_args_existed(current_args, args_list):
            args_list.append(current_args)

    logger.debug("There are {} args_list for evaluating k, l, threshold".format(len(args_list)))
    # evaluate reset_w
    max_l = max(l_values)
    for reset_w, calgo_name in itertools.product(reset_w_values, calgo_names):
        current_args = args.copy()

        new_args = {
            "k": str(max_k),
            "w": str(w),
            "max_dist": str(max_threshold),
            "l": str(max_l),
            "reset_w": str(reset_w),
            "calgo": calgo_name,
        }

        current_args.update(new_args)
        if not check_args_existed(current_args, args_list):
            args_list.append(current_args)

    logger.debug("There are {} args_list for evaluating k, l, threshold, reset_w".format(len(args_list)))

    return args_list

def generate_all_run_mode_args_list(args):
    k_values = get_args_values("k", args)
    logger.debug("k_values: {}".format(k_values))

    l_values = get_args_values("l", args)
    logger.debug("l_values: {}".format(l_values))

    threshold_values = get_args_values("max_dist", args)
    logger.debug("threshold_values: {}".format(threshold_values))

    calgo_names = get_args_values("calgo", args)
    logger.debug("calgo_list: {}".format(calgo_names))

    reset_w_values = get_args_values("reset_w", args)

    args_list = []

    w = -1
    for k, max_dist, l, reset_w, calgo_name in itertools.product(k_values, threshold_values, l_values, reset_w_values, calgo_names):
        current_args = args.copy()
        current_args['k'] = str(k)
        current_args['w'] = str(w)
        current_args['max_dist'] = str(max_dist)
        current_args["l"] = str(l)
        current_args["reset_w"] = str(reset_w)
        current_args["calgo"] = calgo_name

        args_list.append(current_args)

    return args_list

def main(args):
    logger.info(args)

    run_mode = args["run_mode"]

    if run_mode == "normal":
        args_list = generate_normal_run_mode_args_list(args)
    elif run_mode == "all":
        args_list = generate_all_run_mode_args_list(args)
    else:
        raise Exception("Unsupported run mode: {}".format(run_mode))

    logger.debug("There are {} args".format(len(args_list)))
    # find all instances
    file_name = 'anonymize.py'
    Parallel(n_jobs=args['workers'])(
        delayed(rutils.run_python_file)(file_name, args_item)
        for args_item in tqdm(args_list)
    )


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
