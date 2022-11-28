import networkx as nx
import logging
import os

from .dynamic_graph import DynamicGraph

USER_RELATION_NAME = "cite"
GRAPH_NAME = "dblp"

def get_name(prefix, id):
    return "{}-{}".format(prefix, id)

def get_user_name(id):
    return get_name("user", id)

def load_users_relationship(graph, file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda line: line.rstrip().split(" "), lines))

        for u, v in lines:
            graph.add_relationship_edge(get_user_name(u), USER_RELATION_NAME, get_user_name(v), 0)

class DBLPGraph(DynamicGraph):
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_raw_file(data_dir, args):
        # load attributes
        # load relationships
        graph = DBLPGraph()

        user_relationship_path = os.path.join(data_dir, "out.dblp-cite")

        load_users_relationship(graph, user_relationship_path)

        return graph
