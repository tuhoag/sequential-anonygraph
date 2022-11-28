import copy
from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
import glob
import argparse
import logging
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.utils.visualization as vutils
import anonygraph.time_graph_generators as generators
import anonygraph.algorithms.clustering as calgo
import anonygraph.algorithms as algo
import anonygraph.evaluation.clusters_metrics as cmetrics
from anonygraph.constants import *

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_sequence_data_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

    parser.add_argument("--anony_mode")
    parser.add_argument("--refresh", type=rutils.str2bool)

EMAIL_TEMP_NUM_REAL_EDGES = 24929
EMAIL_TEMP_NUM_REAL_ENTITIES = 986
EMAIL_TEMP_NUM_MAX_EDGES = 971210

# def get_email_temp_exp_data_of_cdga():
#     k_list = list(range(1, 11))
#     num_samples = len(k_list)

#     raw_data = {
#         "algo": list(itertools.repeat("cdga", num_samples)),
#         "adm": list(itertools.repeat(None, num_samples)),
#         FAKE_EDGES_METRIC: [0, 942, 1651, 2242, 3137, 3404, 4496, 4883, 5430, 6073],
#         REAL_EDGES_METRIC: list(itertools.repeat(EMAIL_TEMP_NUM_REAL_EDGES, num_samples)),
#         "num_max_edges": list(itertools.repeat(EMAIL_TEMP_NUM_MAX_EDGES, num_samples)),
#         REAL_ENTITIES_METRIC: list(itertools.repeat(EMAIL_TEMP_NUM_REAL_ENTITIES, num_samples)),
#         "k": k_list,
#         "max_dist": list(itertools.repeat(None, num_samples)),
#     }
#     df = pd.DataFrame(raw_data)
#     return df

# def get_email_temp_exp_data_of_dga():
#     k_list = list(range(1, 5))
#     num_samples = len(k_list)

#     raw_data = {
#         "algo": list(itertools.repeat("dga", num_samples)),
#         "adm": list(itertools.repeat(None, num_samples)),
#         FAKE_EDGES_METRIC: [0, 1257, 1683, 4112],
#         REAL_EDGES_METRIC: list(itertools.repeat(EMAIL_TEMP_NUM_REAL_EDGES, num_samples)),
#         "num_max_edges": list(itertools.repeat(EMAIL_TEMP_NUM_MAX_EDGES, num_samples)),
#         REAL_ENTITIES_METRIC: list(itertools.repeat(EMAIL_TEMP_NUM_REAL_ENTITIES, num_samples)),
#         "k": k_list,
#         "max_dist": list(itertools.repeat(None, num_samples)),
#     }

#     df = pd.DataFrame(raw_data)
#     return df

# def get_email_temp_exp_data_of_ckga():
#     k_list = list(range(1, 11))
#     num_samples = len(k_list)

#     raw_data = {
#         "algo": list(itertools.repeat("ckga", num_samples)),
#         "adm": list(itertools.repeat(None, num_samples)),
#         FAKE_EDGES_METRIC: [0, 1345, 2092, 2840, 3878, 4453, 4638, 6195, 6044, 6198],
#         REAL_EDGES_METRIC: list(itertools.repeat(EMAIL_TEMP_NUM_REAL_EDGES, num_samples)),
#         "num_max_edges": list(itertools.repeat(EMAIL_TEMP_NUM_MAX_EDGES, num_samples)),
#         REAL_ENTITIES_METRIC: list(itertools.repeat(EMAIL_TEMP_NUM_REAL_ENTITIES, num_samples)),
#         "k": k_list,
#         "max_dist": list(itertools.repeat(1, num_samples)),
#     }

#     df = pd.DataFrame(raw_data)
#     return df

def get_email_temp_exp_data():
    def get_common_data(num_samples):
        common_data = {
            "data": list(itertools.repeat("email-temp", num_samples)),
            "sample": list(itertools.repeat(-1, num_samples)),
            REAL_ENTITIES_METRIC: list(itertools.repeat(EMAIL_TEMP_NUM_REAL_ENTITIES, num_samples)),
            REAL_EDGES_METRIC: list(itertools.repeat(EMAIL_TEMP_NUM_REAL_EDGES, num_samples)),
            "adm": list(itertools.repeat(None, num_samples)),
            "num_max_edges": list(itertools.repeat(EMAIL_TEMP_NUM_MAX_EDGES, num_samples)),
            RATIO_AVERAGE_CLUSTERING_COEFFICIENT: list(itertools.repeat(None, num_samples)),
            ANONYMIZED_AVERAGE_CLUSTERING_COEFFICIENT: list(itertools.repeat(None, num_samples)),
            RAW_AVERAGE_CLUSTERING_COEFFICIENT: list(itertools.repeat(None, num_samples)),
        }
        return common_data

    k_list = list(range(1, 11))
    num_samples = len(k_list)
    cdga_data = utils.merge_dictionaries(get_common_data(num_samples), {
        "algo": list(itertools.repeat("cdga", num_samples)),
        FAKE_EDGES_METRIC: [0, 942, 1651, 2242, 3137, 3404, 4496, 4883, 5430, 6073],
        "k": k_list,
    })

    ckga_data = utils.merge_dictionaries(get_common_data(num_samples), {
        "algo": list(itertools.repeat("ckga", num_samples)),
        FAKE_EDGES_METRIC: [0, 1345, 2092, 2840, 3878, 4453, 4638, 6195, 6044, 6198],
        "k": k_list,
    })

    k_list = list(range(1, 5))
    num_samples = len(k_list)
    dga_data = utils.merge_dictionaries(get_common_data(num_samples), {
        "algo": list(itertools.repeat("dga", num_samples)),
        FAKE_EDGES_METRIC: [0, 1257, 1683, 4112],
        "k": k_list,
    })

    k_list = [10, 20, 30, 40, 50]
    num_samples = len(k_list)

    kinout_data = utils.merge_dictionaries(get_common_data(num_samples), {
        "algo": list(itertools.repeat("dsndg-kioda", num_samples)),
        FAKE_EDGES_METRIC: list(itertools.repeat(None, num_samples)),
        "k": k_list,
        "acc": [0.06, 0.078, 0.08, 0.09, 0.11]
        # RAW_AVERAGE_CLUSTERING_COEFFICIENT
    })

    df = pd.concat(
        [
            pd.DataFrame(dga_data),
            pd.DataFrame(cdga_data),
            pd.DataFrame(ckga_data),
            pd.DataFrame(kinout_data),
        ]

    )
    return df


# def get_email_temp_exp_data():
#     df = pd.concat([
#         get_email_temp_exp_data_of_cdga(),
#         get_email_temp_exp_data_of_ckga(),
#         get_email_temp_exp_data_of_dga(),
#     ])

#     return df

def get_dblp_exp_data():
    k_list = list(range(1, 11))
    num_samples = len(k_list)

    num_real_edges = 49728
    num_real_entities = 12591
    num_max_edges = 158533281

    real_info = {
        REAL_EDGES_METRIC: list(itertools.repeat(num_real_edges, num_samples)),
        "num_max_edges": list(itertools.repeat(num_max_edges, num_samples)),
        REAL_ENTITIES_METRIC: list(itertools.repeat(num_real_entities, num_samples)),
        "adm": list(itertools.repeat(None, num_samples)),
        "k": k_list,
        "max_dist": list(itertools.repeat(1, num_samples)),
    }

    ckga_data = utils.merge_dictionaries(real_info, {
        "algo": list(itertools.repeat("ckga", num_samples)),
        FAKE_EDGES_METRIC: [0, 1768, 2470, 3924, 4535, 5904, 6555, 7547, 8717, 9085],
    })

    cdga_data = utils.merge_dictionaries(real_info,{
        "algo": list(itertools.repeat("cdga", num_samples)),
        FAKE_EDGES_METRIC: [0, 1289, 1895, 2365, 4316, 4316, 4718, 5399, 6513, 9815],
    })

    dga_data = utils.merge_dictionaries(real_info, {
        "algo": list(itertools.repeat("dga", num_samples)),
        FAKE_EDGES_METRIC: [0, 816, 1908, 2516, 3252, 4157, 4627, 5552, 6521, 7022],
    })

    df = pd.concat([
        pd.DataFrame(dga_data),
        pd.DataFrame(ckga_data),
        pd.DataFrame(cdga_data),
    ])
    return df

def get_freebase_exp_data():
    k_list = list(range(1, 11))
    num_samples = len(k_list)

    num_real_edges = 43780
    num_real_entities = 5000
    num_max_edges = 0

    real_info = {
        REAL_EDGES_METRIC: list(itertools.repeat(num_real_edges, num_samples)),
        "num_max_edges": list(itertools.repeat(num_max_edges, num_samples)),
        REAL_ENTITIES_METRIC: list(itertools.repeat(num_real_entities, num_samples)),
        FAKE_EDGES_METRIC: list(itertools.repeat(None, num_samples)),
        "k": k_list,
        "max_dist": list(itertools.repeat(1, num_samples)),
    }

    ckga_data = utils.merge_dictionaries(real_info, {
        "algo": list(itertools.repeat("ckga", num_samples)),
        "calgo": list(itertools.repeat("km", num_samples)),
        "adm": [0, 0.001614685, 0.003065127, 0.00442576, 0.005404223, 0.006403854, 0.007907353, 0.008982707, 0.009607287, 0.010879092],
    })

    return pd.DataFrame(ckga_data)

def get_email_exp_data():
    k_list = list(range(2, 11))
    num_samples = len(k_list)

    num_real_edges = 26576
    num_real_entities = 1005
    num_max_edges = 0

    real_info = {
        REAL_EDGES_METRIC: list(itertools.repeat(num_real_edges, num_samples)),
        "num_max_edges": list(itertools.repeat(num_max_edges, num_samples)),
        REAL_ENTITIES_METRIC: list(itertools.repeat(num_real_entities, num_samples)),
        FAKE_EDGES_METRIC: list(itertools.repeat(None, num_samples)),
        "k": k_list,
        "max_dist": list(itertools.repeat(1, num_samples)),
    }

    ckga_data = utils.merge_dictionaries(real_info, {
        "algo": list(itertools.repeat("ckga", num_samples)),
        "calgo": list(itertools.repeat("km", num_samples)),
        "adm": [0.013948327, 0.02184327, 0.024901294, 0.028136367, 0.038808165, 0.044123674, 0.048523348, 0.047574248, 0.05351249],
    })

    return pd.DataFrame(ckga_data)

def get_previous_works_exp_data(data_name):
    data_fn_dict = {
        "email-temp": get_email_temp_exp_data,
        "dblp": get_dblp_exp_data,
        "freebase": get_freebase_exp_data,
        "email": get_email_exp_data,
    }
    data_fn = data_fn_dict.get(data_name)

    if data_fn is not None:
        df = data_fn()
    else:
        raise Exception("Unsupported data: {}".format(data_name))

    return df

def prepare_data(data_info, num_workers, args):
    anony_mode = data_info["anony_mode"]

    if anony_mode == CLUSTERS_ANONYMIZATION_MODE:
        raw_data = vutils.prepare_clusters_data(data_info, num_workers, args)
    elif anony_mode == CLUSTERS_AND_GRAPH_ANONYMIZATION_MODE:
        raw_data = vutils.prepare_anonymized_subgraphs_data(data_info, num_workers, args)
    else:
        raise Exception("Unsupported anony mode: {}".format(anony_mode))

    return raw_data


def add_more_info(df):
    df["ratio_fake_edges"] = df[FAKE_EDGES_METRIC
                               ] / df[REAL_EDGES_METRIC]
    # df["ratio_fake_entities"] = df[FAKE_ENTITIES_METRIC
    #                               ] / df[REAL_ENTITIES_METRIC]
    # df["acc"] = abs(df[ANONYMIZED_AVERAGE_CLUSTERING_COEFFICIENT] - df[RAW_AVERAGE_CLUSTERING_COEFFICIENT]) / df[RAW_AVERAGE_CLUSTERING_COEFFICIENT]

    df[r"$\tau$"] = df["max_dist"]
    df["algo_name"] = df["algo"]
    df.loc[df["algo"]!= "skga", "l"] = 1
    df.loc[df["algo"]!= "skga", "w"] = -1
    df.loc[df["algo"]!= "skga", "reset_w"] = -1

    df.loc[df["algo"] == "skga", "algo"] = "takga"


def visualize_comparision(
    df, w_values, reset_w_values, k_values, l_values, max_dist_values, calgo_names, algo_values, y_name, x_name, cat_name, path=None
):
    logger.debug(df)

    # logger.debug(df["algo"].unique())
    df.sort_values(by=["algo", "k"], inplace=True)
    df = df[
        (df["k"].isin(k_values))
        & (df["l"].isin(l_values))
        & (df["algo"].isin(algo_values))
        & (df["max_dist"].isin(max_dist_values))
        & (df["calgo"].isin(calgo_names))
        # & (df["reset_w"].isin(reset_w_values))
        # & (df["w"].isin(w_values))
        # & (df["w"].isin(w_values)))
    ]

    df[y_name] = df[y_name].apply(pd.to_numeric)

    logger.info("visualizing data: {}".format(df))
    x_values = df[x_name].unique()
    cat_values = df[cat_name].unique()
    y_values = df[y_name].unique()

    logger.debug("{} values: {}".format(x_name, x_values))
    logger.debug("{} values: {}".format(cat_name, cat_values))
    logger.debug("{} values: {}".format(y_name, y_values))

    logger.debug("filtered df (len: {}): {}".format(len(df), df[[x_name, cat_name, y_name, "max_dist", "calgo", "l"]]))
    num_cat_values = len(df[cat_name].unique())

    for cat_value in cat_values:
        current_df = df[df[cat_name].isin([cat_value])]
        logger.debug("cat value: {} - data: {}".format(cat_value, current_df[[x_name, y_name, cat_name]]))

    temp_df = df.groupby(by=cat_name)
    logger.debug("temp df: {}".format(temp_df[[x_name, y_name, cat_name]]))
    current_palette = sns.color_palette(n_colors=num_cat_values, palette="bright")
    sns.lineplot(
        data=df, x=x_name, y=y_name,
        hue=cat_name,
        style=cat_name,
        palette=current_palette,
        markers=True,
    )

    plt.ylabel(vutils.get_title(y_name))
    plt.grid(linestyle="--")
    plt.xticks(vutils.get_xticks(x_name, x_values, 8))
    # plt.ylim([0, 0.06])
    plt.legend(title=vutils.get_title(cat_name), labels=cat_values)

    if path is not None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        plt.savefig(path)
        logger.info("saved to {}".format(path))

    plt.show()
    plt.clf()

def main(args):
    logger.debug(args)
    args["strategy"] = STATIC_STRATEGY
    data_name = args["data"]
    sample = args["sample"]
    strategy = args["strategy"]
    anony_mode = args["anony_mode"]
    sattr_name = dutils.get_sensitive_attribute_name(data_name, args["sattr"])

    data_path = putils.get_comparision_exp_data_path(
        data_name, sample, strategy, anony_mode, args
    )

    our_df = vutils.get_exp_data(
        exp_path=data_path,
        prepare_data_fn=prepare_data,
        prepare_data_args={
            "data": data_name,
            "sample": sample,
            "strategy": strategy,
            "anony_mode": anony_mode,
        },
        workers=args["workers"],
        refresh=args["refresh"],
        args=args
    )

    other_df = get_previous_works_exp_data(data_name)
    logger.debug("other df (len: {}): {}".format(len(other_df), other_df))

    df = pd.concat([our_df, other_df])
    logger.debug("loaded df (len: {}): {}".format(len(df), df))
    add_more_info(df)

    df.to_csv("temp.csv")
    # raise Exception()
    logger.info("visualizing")
    if data_name in ["email", "freebase"]:
        metric_names = [ADM_METRIC]
    elif data_name in ["dblp", "email-temp"]:
        metric_names = ["ratio_fake_edges"]
    else:
        raise Exception("Unsupported data: {}".format(data_name))

    fig_dir_path = os.path.join(os.path.dirname(data_path), "figures")
    for metric_name in metric_names:
        fig_path = os.path.join(fig_dir_path, "compare-{}#{}#{}-{}-k.pdf".format(data_name, sattr_name, strategy, metric_name))
        visualize_comparision(
            df=df,
            w_values=[1],
            reset_w_values=[-1],
            calgo_names=["km"],
            k_values=list(range(2, 11)),
            l_values=[1],
            max_dist_values=[1],
            algo_values = ["cdga", "dga", "ckga", "takga"],
            x_name="k",
            y_name=metric_name,
            cat_name="algo",
            path=fig_path,
        )

        # fig_path = os.path.join(fig_dir_path, "compare-{}-{}-tau.pdf".format(data_name, metric_name))
        # visualize_comparision(
        #     df=df,
        #     w_values=[1],
        #     reset_w_values=[-1],
        #     calgo_names=["km"],
        #     k_values=[10],
        #     max_dist_values=[0, 0.25, 0.5, 0.75, 1],
        #     algo_values = ["cdga", "dga", "ckga", "skga"],
        #     x_name="max_dist",
        #     y_name=metric_name,
        #     cat_name="algo",
        #     path=fig_path,
        # )

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
