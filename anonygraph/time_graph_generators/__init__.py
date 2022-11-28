from .period_generator import PeriodGenerator
from .equal_addition_size_generator import EqualAdditionSizeGenerator
from .equal_raw_generator import EqualRawSizeGenerator
from .raw_generator import RawGenerator
from .raw_addition_generator import RawAdditionGenerator
from .mean_addition_edges_generator import MeanAdditionEdgesStrategy
from .static_generator import StaticGenerator
from .mean_edges_generator import MeanEdgesStrategy

from anonygraph.constants import *

def get_strategy(strategy_name, args):
    if strategy_name == RAW_STRATEGY:
        gen = RawGenerator(args["workers"])
    elif strategy_name == PERIOD_GEN_STRATEGY:
        gen = PeriodGenerator(args["period"], args["unit"])
    elif strategy_name == EQUAL_ADDITION_SIZE_STRATEGY:
        gen = EqualAdditionSizeGenerator(
            args["n_sg"], args["workers"]
        )
    elif strategy_name == EQUAL_RAW_SIZE_STRATEGY:
        gen = EqualRawSizeGenerator(args["n_sg"], args["workers"])
    elif strategy_name == RAW_ADDITION_STRATEGY:
        gen = RawAdditionGenerator(args["n_sg"], args["workers"])
    elif strategy_name == MEAN_ADDITION_EDGES_STRATEGY:
        gen = MeanAdditionEdgesStrategy(args["n_sg"], args["workers"])
    elif strategy_name == STATIC_STRATEGY:
        gen = StaticGenerator(args["workers"])
    elif strategy_name == MEAN_EDGES_STRATEGY:
        gen = MeanEdgesStrategy(args["n_sg"], args["workers"])
    else:
        raise NotImplementedError(
            "Unsupported strategy: {}".format(strategy_name)
        )

    return gen
