import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import DGLDataset
from anonygraph.evaluation.classification.data_loader import DummyGraph, AnonyGraph
from anonygraph.evaluation.classification.rgcn_model import RGCN
from anonygraph.evaluation.classification.trainer import train
import anonygraph.utils.path as putils
import anonygraph.utils.runner as rutils
import anonygraph.utils.visualization as vutils
import warnings
import json
import os
import itertools

logger = logging.getLogger(__name__)

warnings.filterwarnings(action='ignore', category=UserWarning)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_sequence_data_argument(parser)

    parser.add_argument("--refresh", type=rutils.str2bool, default="n")
    parser.add_argument("--workers", type=int, default=1)
    # rutils.add_graph_generalization_argument(parser)

    # parser.add_argument('--d_list', type=rutils.string2list(int))
    # parser.add_argument('--k_list', type=rutils.string2list(int))
    # parser.add_argument('--w_list', type=rutils.string2list(int))
    # parser.add_argument('--max_dist_list', type=rutils.string2list(float))
    # parser.add_argument('--l_list', type=rutils.string2list(int))
    # parser.add_argument("--calgo_list", type=rutils.string2list(str))
    # parser.add_argument("--reset_w_list", type=rutils.string2list(int))

    # rutils.add_workers_argument(parser)

    rutils.add_log_argument(parser)
    # parser.add_argument("--refresh", type=rutils.str2bool)


def get_metric_data_over_time(data):
    t_list = sorted(map(int, data.keys()))
    logger.debug(t_list)

    results = []
    metric_name = "test_avg_accuracy"

    for t in t_list:
        t_data = data.get(str(t))
        metric_data = t_data[metric_name]
        best_results = metric_data["best_results"]

        avg_result = np.mean(sorted(best_results))
        results.append(avg_result)

    return t_list,results

def visualize_over_k(data, d, k_list):
    logger.debug(data.keys())
    model_data = data[str(d)]

    # visualize_accuracy
    # key = "{}_{}_{}_{}_{}".format(k,l,reset_w,calgo,enforcer_str)
    raw_data = model_data.get("raw", None)
    # t_list = sorted(map(int, data.keys()))
    t_list,raw_results = get_metric_data_over_time(raw_data)
    # logger.debug("{}:{}".format(len(t_list), t_list))
    # logger.debug("{}:{}".format(len(raw_results), raw_results))
    plt.plot(t_list, raw_results, label="raw")

    l = 1
    reset_w = -1
    calgo="km"
    enforcer_str="gs#1.00"
    for k in k_list:
        key = "{}_{}_{}_{}_{}".format(k,l,reset_w,calgo,enforcer_str)
        anony_data = model_data.get(key, None)
        # logger.debug(anony_data)
        if anony_data is None:
            continue

        t_list,anony_results = get_metric_data_over_time(anony_data)
        plt.plot(t_list, anony_results, label=k)

    plt.legend()
    plt.xticks(t_list)
    plt.savefig("test-k.png")
    plt.show()

def visualize_over_l(data, d, l_list):
    logger.debug(data.keys())
    model_data = data[str(d)]

    # visualize_accuracy
    # key = "{}_{}_{}_{}_{}".format(k,l,reset_w,calgo,enforcer_str)
    raw_data = model_data.get("raw", None)
    # t_list = sorted(map(int, data.keys()))
    t_list,raw_results = get_metric_data_over_time(raw_data)
    # logger.debug("{}:{}".format(len(t_list), t_list))
    # logger.debug("{}:{}".format(len(raw_results), raw_results))
    plt.plot(t_list, raw_results, label="raw")

    k = 10
    reset_w = -1
    calgo="km"
    enforcer_str="gs#1.00"
    for l in l_list:
        key = "{}_{}_{}_{}_{}".format(k,l,reset_w,calgo,enforcer_str)
        anony_data = model_data.get(key, None)
        # logger.debug(anony_data)
        if anony_data is None:
            continue

        t_list,anony_results = get_metric_data_over_time(anony_data)
        plt.plot(t_list, anony_results, label=l)

    plt.legend()
    plt.xticks(t_list)
    plt.savefig("test-l.png")
    plt.show()


def visualize_line_chart(df, x_name, y_name, cat_name, path):
    x_values = df[x_name].unique()
    cat_values= df[cat_name].unique()

    logger.debug("x: {} - values: {}".format(x_name, x_values))
    logger.debug("cat: {} - values: {}".format(cat_name, cat_values))

    # sns.set_palette("pastel")
    custom_palette = sns.color_palette("bright", len(cat_values))
    sns.set_palette(custom_palette)
    # sns.palplot(custom_palette)

    # logger.debug(df)
    figure = sns.lineplot(data=df, y=y_name, x=x_name, hue=cat_name, style=cat_name, palette=custom_palette, markers=True).get_figure()

    plt.ylabel(get_title(y_name))
    plt.xlabel(get_title(x_name))
    plt.grid(linestyle="--", axis="y", color="grey", linewidth=0.5)
    plt.xticks(x_values)
    plt.legend(title=get_title(cat_name))

    if path is not None:
        save_figure(figure, path)

    plt.show()
    plt.clf()

def save_figure(figure, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    logger.info("saving figure to: {}".format(path))
    figure.savefig(path)

def visualize(df, x_name, cat_name, k_values, reset_w_values, l_values, enforcer_values, calgo_values, prefix):
    df.sort_values(by=["calgo", "enforcer_str", "k", "l", "reset_w", "max_dist", "t"], inplace=True)
    d_values=[200]
    logger.debug(df["data"].unique())
    data_name = df["data"].unique()[0]

    logger.debug(df["k"].unique())
    logger.debug(df["l"].unique())
    logger.debug(df["reset_w"].unique())
    logger.debug(df["calgo"].unique())
    logger.debug(df["enforcer_str"].unique())
    logger.debug(df["d"].unique())
    logger.debug(df["top-5_test_avg_accuracy"].unique())
    # logger.debug(df.columns)

    df = df[(df["k"].isin(k_values))
            & (df["l"].isin(l_values))
            & (df["d"].isin(d_values))
            & (df["reset_w"].isin(reset_w_values))
            # & (df["enforcer_str"].isin(enforcer_values))
            & (df["calgo"].isin(calgo_values))
            ]
    # df = df[df["k"] == 1]

    logger.debug(df[[x_name, "key_l", "key_k","train_avg_f1"]])
    metrics = [
        # "train_avg_f1",
        # "train_avg_accuracy",
        "test_avg_accuracy",
        "top-5_test_avg_accuracy",
    ]

    for metric in metrics:
        path = os.path.join("exp_data", "training", "{}_{}_{}.png".format(prefix,data_name,metric))
        logger.debug(metric)
        visualize_line_chart(
            df=df,
            x_name=x_name,
            y_name=metric,
            cat_name=cat_name,
            path=path,
        )

def get_title(name):
    name2title = {
        "key_k": "k",
        "key_l": "l",
        "test_avg_accuracy": "Accuracy (%)"
    }

    title = name2title.get(name)

    if title is None:
        title = name

    return title

def summarize_results(df, cat_name, k_values, reset_w_values, l_values, enforcer_values, calgo_values):
    df.sort_values(by=["calgo", "enforcer_str", "k", "l", "reset_w", "max_dist", "t"], inplace=True)
    d_values=[200]
    # logger.debug(df["data"].unique())
    # data_name = df["data"].unique()[0]

    # logger.debug(df["k"].unique())
    # logger.debug(df["l"].unique())
    # logger.debug(df["reset_w"].unique())
    # logger.debug(df["calgo"].unique())
    # logger.debug(df["enforcer_str"].unique())
    # logger.debug(df["d"].unique())
    # logger.debug(df["top-5_test_avg_accuracy"].unique())
    # # logger.debug(df.columns)

    df = df[(df["k"].isin(k_values))
            & (df["l"].isin(l_values))
            & (df["d"].isin(d_values))
            & (df["reset_w"].isin(reset_w_values))
            # & (df["enforcer_str"].isin(enforcer_values))
            & (df["calgo"].isin(calgo_values))
            ]

    temp = df.groupby(by=cat_name).mean("top-5_test_avg_accuracy")
    logger.info(temp[["top-5_test_avg_accuracy", "test_avg_accuracy"]])

def add_more_info(df):
    df["key_k"] = df["k"]
    df["key_k"].replace(1, "raw", inplace=True)

    df["key_l"] = df["l"]
    df.loc[df["key_k"] == "raw", "key_l"] = "raw"
    # df["key_l"].replace("1", "raw", inplace=True)

    logger.debug(df["key_k"].unique())
    logger.debug(df["key_l"].unique())


def main(args):
    data_name = args["data"]
    strategy_name = args["strategy"]

    data_path = putils.get_agg_training_data_path(data_name, strategy_name, args)
    df = vutils.get_exp_data(
        exp_path=data_path,
        prepare_data_fn=vutils.prepare_training_data,
        prepare_data_args={
            "data": args["data"],
            "sample": args["sample"],
            "strategy": args["strategy"],
        },
        workers=args["workers"],
        refresh=args["refresh"],
        args=args
    )

    add_more_info(df)
    # data = vutils.load_training_data(args, "")


    # data = vutils.load_training_data2(args)

    # df = pd.DataFrame(data)

    # logger.debug(df)
    # logger.debug(data_copy.keys())
    # logger.debug(data.keys())




    visualize(
        df=df,
        x_name="t",
        cat_name="key_k",
        k_values=[1, 2,4,6,8,10],
        l_values=[1],
        reset_w_values=[-1],
        calgo_values=["km", "raw"],
        enforcer_values=["gs#1.00", "raw"],
        prefix="k",
    )

    visualize(
        df=df,
        x_name="t",
        cat_name="key_l",
        k_values=[1,10],
        l_values=[1,2,3],
        reset_w_values=[-1],
        calgo_values=["km", "raw"],
        enforcer_values=["gs#1.00", "raw"],
        prefix="l",
    )

    summarize_results(
        df=df,
        # x_name="t",
        cat_name="key_k",
        k_values=[1, 2,4,6,8,10],
        l_values=[1],
        reset_w_values=[-1],
        calgo_values=["km", "raw"],
        enforcer_values=["gs#1.00", "raw"],
    )

    summarize_results(
        df=df,
        # x_name="t",
        cat_name="key_l",
        k_values=[1,10],
        l_values=[1,2,3,4],
        reset_w_values=[-1],
        calgo_values=["km", "raw"],
        enforcer_values=["gs#1.00", "raw"],
    )
    # visualize_over_k(data, d=200, k_list=[2,4,6,8,10])
    # visualize_over_l(data, d=200, l_list=[1,2,3])

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)

