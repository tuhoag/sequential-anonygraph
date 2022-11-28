import networkx as nx
import logging
import os

from .dynamic_graph import DynamicGraph
from .freebase_graph import load_static_graph_from_old_output

logger = logging.getLogger(__name__)

class EmailGraph(DynamicGraph):
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_raw_file(data_dir, args):
        graph = EmailGraph()
        load_static_graph_from_old_output(data_dir, graph, "user", "", args)
        return graph
