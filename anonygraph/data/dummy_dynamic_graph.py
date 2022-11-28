import networkx as nx
import logging
import os


from .dynamic_graph import DynamicGraph

logger = logging.getLogger(__name__)

GRAPH_NAME = 'email'
USER_RELATION_NAME = 'sent'
ATTR_RELATION_NAME = 'belongs_to'

class DummyDynamicGraph(DynamicGraph):
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_raw_file(data_dir, args):
        # load attributes
        # load relationships
        graph = DynamicGraph()

        t = 1
        graph.add_attribute_edge('user_0', 'age', 21, t)
        graph.add_attribute_edge('user_0', 'job', 'Student', t)
        graph.add_relationship_edge('user_0', 'follows', 'user_2', t)
        graph.add_attribute_edge("user_0", "disease", "flu", t)

        graph.add_attribute_edge('user_1', 'age', 19, t)
        graph.add_attribute_edge('user_1', 'job', 'Student', t)
        # graph.add_relationship_edge('user_1', 'follows', 'user_3', t)
        graph.add_attribute_edge("user_1", "disease", "gast.", t)

        graph.add_attribute_edge('user_2', 'age', 21, t)
        graph.add_attribute_edge('user_2', 'job', 'Engineer', t)
        graph.add_attribute_edge("user_2", "disease", "flu", t)

        graph.add_attribute_edge('user_3', 'age', 30, t)
        graph.add_attribute_edge('user_3', 'job', 'Engineer', t)
        graph.add_attribute_edge("user_3", "disease", "dysp.", t)

        graph.add_attribute_edge('user_4', 'age', 30, t)
        graph.add_attribute_edge('user_4', 'job', 'Engineer', t)
        # graph.add_relationship_edge('user_4', 'follows', 'user_5', t)
        graph.add_attribute_edge("user_4", "disease", "bron.", t)

        graph.add_attribute_edge('user_5', 'job', 'Engineer', t)
        graph.add_attribute_edge("user_5", "disease", "flu", t)

        t = 2
        graph.add_attribute_edge('user_0', 'age', 21, t)
        graph.add_attribute_edge('user_0', 'job', 'Student', t)
        graph.add_relationship_edge('user_0', 'follows', 'user_2', t)
        graph.add_attribute_edge("user_0", "disease", "flu", t)

        # remove user 1
        # graph.add_attribute_edge('user_1', 'age', 19, t)
        # graph.add_attribute_edge('user_1', 'job', 'Student', t)
        # graph.add_attribute_edge("user_1", "disease", "gast.", t)

        graph.add_attribute_edge('user_2', 'age', 21, t)
        graph.add_attribute_edge('user_2', 'job', 'Engineer', t)
        graph.add_attribute_edge("user_2", "disease", "flu", t)

        graph.add_attribute_edge('user_3', 'age', 30, t)
        graph.add_attribute_edge('user_3', 'job', 'Engineer', t)
        graph.add_attribute_edge("user_3", "disease", "dysp.", t)

        graph.add_attribute_edge('user_4', 'age', 30, t)
        graph.add_attribute_edge('user_4', 'job', 'Engineer', t)
        graph.add_attribute_edge("user_4", "disease", "bron.", t)

        # update class of user 5
        graph.add_attribute_edge('user_5', 'job', 'Engineer', t)
        graph.add_attribute_edge("user_5", "disease", "bron.", t)
        # raise Exception()
        # new user which should be removed
        graph.add_attribute_edge("user_6", "age", 19, t)
        graph.add_attribute_edge("user_6", "job", "Student", t)
        graph.add_attribute_edge("user_6", "disease", "bron.", t)

        t = 3
        graph.add_attribute_edge('user_0', 'age', 21, t)
        graph.add_attribute_edge('user_0', 'job', 'Student', t)
        graph.add_relationship_edge('user_0', 'follows', 'user_2', t)
        graph.add_attribute_edge("user_0", "disease", "flu", t)

        # re-inserted user
        graph.add_attribute_edge('user_1', 'age', 19, t)
        graph.add_attribute_edge('user_1', 'job', 'Student', t)
        graph.add_attribute_edge("user_1", "disease", "gast.", t)

        graph.add_attribute_edge('user_2', 'age', 21, t)
        graph.add_attribute_edge('user_2', 'job', 'Engineer', t)
        graph.add_attribute_edge("user_2", "disease", "flu", t)

        graph.add_attribute_edge('user_3', 'age', 30, t)
        graph.add_attribute_edge('user_3', 'job', 'Engineer', t)
        graph.add_attribute_edge("user_3", "disease", "dysp.", t)

        # remove user 4
        # graph.add_attribute_edge('user_4', 'age', 30, t)
        # graph.add_attribute_edge('user_4', 'job', 'Engineer', t)
        # graph.add_attribute_edge("user_4", "disease", "bron.", t)

        # remove user 5
        # graph.add_attribute_edge('user_5', 'job', 'Engineer', t)
        # graph.add_attribute_edge("user_5", "disease", "bron.", t)

        # new user which is not published at t=2
        graph.add_attribute_edge("user_6", "age", 19, t)
        graph.add_attribute_edge("user_6", "job", "Student", t)
        graph.add_attribute_edge("user_6", "disease", "bron.", t)

        # new user at t=3
        graph.add_attribute_edge("user_7", "age", 21, t)
        graph.add_attribute_edge("user_7", "job", "Student", t)
        graph.add_attribute_edge("user_7", "disease", "flu", t)

        return graph
