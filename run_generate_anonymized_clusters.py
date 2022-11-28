import argparse
import logging
import os
import itertools
from joblib import Parallel, delayed

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.time_graph_generators as generators
import anonygraph.algorithms.clustering as calgo

logger = logging.getLogger(__file__)


def add_arguments(parser):
    parser.add_argument('--data')
    parser.add_argument('--sample')

    parser.add_argument('--strategy')
    parser.add_argument('--period')
    parser.add_argument('--unit')
    parser.add_argument('--n_sg')

    parser.add_argument('--info_loss')
    parser.add_argument('--alpha_adm')
    parser.add_argument('--alpha_dm')

    parser.add_argument('--k_range', type=rutils.string2range(int))
    parser.add_argument('--k_list', type=rutils.string2list(int))
    parser.add_argument('--w_range', type=rutils.string2range(int))
    parser.add_argument('--w_list', type=rutils.string2list(int))
    parser.add_argument('--max_dist_list', type=rutils.string2list(float))

    parser.add_argument('--calgo')


    parser.add_argument('--workers', type=int)

    rutils.add_log_argument(parser)


def main(args):
    logger.info(args)

    if args['k_list'] is not None and args['k_range'] is not None:
        raise Exception("k_list and k_range cannot be passed at the same time.")
    else:
        k_range = args['k_list'] if args['k_list'] is not None else args['k_range']

    logger.debug('k range: {}'.format(k_range))

    if args['w_list'] is not None and args['w_range'] is not None:
        raise Exception("w_list and w_range cannot be passed at the same time.")
    else:
        w_range = args['w_list'] if args['w_list'] is not None else args['w_range']

    logger.debug('w range: {}'.format(w_range))
    max_dist_range = args['max_dist_list']

    args_list = []


    for k, w, max_dist in itertools.product(k_range, w_range, max_dist_range):
        current_args = args.copy()
        current_args['k'] = str(k)
        current_args['w'] = str(w)
        current_args['max_dist'] = str(max_dist)

        args_list.append(current_args)

    logger.debug(args_list)
    # find all instances
    # parts = [('generate_anonymous_clusters.py', args_item) for args_item in args_list]
    file_name = 'generate_anonymized_clusters.py'
    Parallel(n_jobs=args['workers'])(delayed(rutils.run_python_file)(file_name, args_item) for args_item in args_list)
    # Parallel(n_job=args['n_workers'])(delayed(temp)(part for part in parts))
    # with multiprocessing.Pool(args['workers']) as p:
    #     p.starmap(utils.run_python_file, parts)

def temp(a, b):
    print("a = {}, b={}".format(a, b))
    return 1

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
