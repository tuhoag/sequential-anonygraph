import os
from anonygraph.constants import CONSOLE_LOG_MODE, FILE_LOG_MODE
import subprocess
import numpy as np
import logging
import argparse

logger = logging.getLogger(__name__)

def setup_arguments(add_arguments_fn):
    parser = argparse.ArgumentParser(description="Process some integers.")

    add_arguments_fn(parser)

    args, _ = parser.parse_known_args()

    params = {}
    for arg in vars(args):
        params[arg] = getattr(args, arg)

    return params


class string2list():
    def __init__(self, out_type):
        self.type = out_type

    def __call__(self, value):
        return [self.type(item) for item in value.split(",")]


class string2range():
    def __init__(self, out_type):
        self.type = out_type

    def __call__(self, value):
        start, stop, interval = map(self.type, value.split(","))
        output = np.arange(start, stop, interval)

        return output

def str2bool(value):
    if value is None:
        return None

    if value in ["yes", "True", "y"]:
        return True
    elif value in ["no", "False", "n"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def str2log_mode(value):
    if value is None:
        return None

    if value in ["d", "debug", "10"]:
        log_mode = logging.DEBUG
        # return logging.DEBUG
    elif value in ["i", "info", "20"]:
        log_mode = logging.INFO
    elif value in ["w", "warning", "30"]:
        log_mode = logging.WARNING
    else:
        raise argparse.ArgumentTypeError("Unsupported log mode type: {}".format(value))

    return log_mode

def add_log_argument(parser):
    parser.add_argument("--log", type=str2log_mode, default=logging.INFO)
    parser.add_argument("--log_modes", type=string2list(str), default=[CONSOLE_LOG_MODE])

def setup_console_logging(args):
    level = args["log"]
    log_modes = args["log_modes"]


    logger = logging.getLogger("")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(name)-12s: %(funcName)s(%(lineno)d) %(levelname)-8s %(message)s"
    )

    for mode in log_modes:
        if mode == CONSOLE_LOG_MODE:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        elif mode == FILE_LOG_MODE:
            if os.path.exists("log.txt"):
                os.remove("log.txt")
            file_handler = logging.FileHandler("log.txt")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        else:
            raise Exception("Unsupported log mode: {}".format(mode))

def add_data_argument(parser):
    parser.add_argument("--data")
    parser.add_argument("--sample", type=int, default=-1)

def add_sequence_data_argument(parser):
    parser.add_argument("--strategy")
    parser.add_argument("--period", type=int)
    parser.add_argument("--unit")
    parser.add_argument("--n_sg", type=int)
    parser.add_argument("--sattr")

def add_info_loss_argument(parser):
    parser.add_argument("--info_loss", default="adm")
    parser.add_argument("--alpha_adm", type=float, default=0.5)
    parser.add_argument("--alpha_dm", type=float, default=0.5)

def add_constraint_argument(parser):
    parser.add_argument("--k", type=int)
    parser.add_argument("--w", type=int, default=-1)
    parser.add_argument("--l", type=int)

def add_clustering_argument(parser):
    parser.add_argument("--calgo")

def add_workers_argument(parser):
    parser.add_argument("--workers", type=int, default=1)

def add_enforcer_argument(parser):
    parser.add_argument("--enforcer")
    parser.add_argument("--max_dist", type=float)

def add_clusters_generation_argument(parser):
    add_data_argument(parser)
    add_sequence_data_argument(parser)
    add_info_loss_argument(parser)
    add_constraint_argument(parser)
    add_clustering_argument(parser)
    add_enforcer_argument(parser)
    parser.add_argument("--min_t", type=int, default=0)
    parser.add_argument("--max_t", type=int, default=-1)
    parser.add_argument("--t", type=int)
    parser.add_argument("--anony_mode", required=True)
    parser.add_argument("--reset_w", type=int, default=-1)

def add_generalization_argument(parser):
    parser.add_argument("--galgo")

def add_graph_generalization_argument(parser):
    add_clusters_generation_argument(parser)

    add_generalization_argument(parser)

def run_python_file(path, args):
    arguments = ["python", path]
    for name, value in args.items():
        if value is not None:
            arguments.append("--" + name)

            if type(value) is list:
                logger.debug(value)
                arguments.append(",".join(map(str, value)))
            else:
                arguments.append(str(value))


    logger.debug("run {}: {}".format(path, arguments))

    with subprocess.Popen(arguments, stdout=subprocess.PIPE) as process:
        for line in iter(process.stdout):
            logger.debug(line.rstrip().decode("utf-8"))

        process.communicate()
        return process.returncode

def run_generate_svals_file(args):
    run_python_file("generate_svals.py", args)

def run_generate_time_groups_file(args):
    run_python_file("generate_time_groups.py", args)


def run_generate_raw_subgraph_file(dest_t, args):
    args["dest_t"] = dest_t

    run_python_file("generate_raw_subgraph.py", args)

def run_generate_all_raw_subgraphs_file(args):
    run_python_file("generate_raw_subgraphs.py", args)

def run_anonymization_runner(args):
    run_python_file("run_anonymization.py", args)

