import logging

logger = logging.getLogger(__name__)

from .base_info_loss import BaseInfoLossMetric
from .info import get_generalized_degree_info

def calculate_for_each_entity(subgraph, entity_ids, degree_type):
    result = {}
    for entity_id in entity_ids:
        result[entity_id] = 0.0

    if len(entity_ids) == 0:
        return result

    num_entities = subgraph.num_entities
    num_relationships = subgraph.num_relationship_relations

    union_info, entities_info = get_generalized_degree_info(subgraph, entity_ids, degree_type)

    logger.debug('union: {} - entities ({}): {} '.format(union_info, entity_ids, entities_info))
    if len(union_info) != 0:
        # num_real_entities = 0
        for entity_id, entity_info in entities_info.items():
            if subgraph.is_entity_id(entity_id):
                # num_real_entities += 1
                entity_score = 0.0

                for relation_id, union_degree in union_info.items():
                    entity_degree = entity_info.get(relation_id, 0)

                    current_score = (union_degree - entity_degree) / (num_entities)

                    entity_score += current_score
                    logger.debug('relation: {} union: {} - info: {} - max: {} - score: {}'.format(relation_id, union_degree, entity_degree, num_entities, current_score))

                entity_score = entity_score / num_relationships
                logger.debug('cluster: {} - user: {} - score: {}'.format(entity_ids, entity_info, entity_score / num_relationships))
            # print(score)
            else:
                entity_score = 1

            result[entity_id] = entity_score

    return result

def calculate_degree_info_loss(subgraph, entity_ids, degree_type):
    result = calculate_for_each_entity(subgraph, entity_ids, degree_type)
    score = sum(result.values()) / len(entity_ids)

    return score

class OutDegreeInfoLoss(BaseInfoLossMetric):
    def calculate_for_each_entity(self, entity_ids):
        return calculate_for_each_entity(self.subgraph, entity_ids, 'out')

    def call(self, entity_ids):
        return calculate_degree_info_loss(self.subgraph, entity_ids, 'out')

class InDegreeInfoLoss(BaseInfoLossMetric):
    def calculate_for_each_entity(self, entity_ids):
        return calculate_for_each_entity(self.subgraph, entity_ids, 'in')

    def call(self, entity_ids):
        return calculate_degree_info_loss(self.subgraph, entity_ids, 'in')

class OutInDegreeInfoLoss(BaseInfoLossMetric):
    def __init__(self, graph, args):
        super().__init__(graph, args)
        self.alpha_dm = args['alpha_dm']
        self.out_degree_info_loss_fn = OutDegreeInfoLoss(graph, args)
        self.in_degree_info_loss_fn = InDegreeInfoLoss(graph, args)

    def calculate_for_each_entity(self, entity_ids):
        result = {}

        out_result = self.out_degree_info_loss_fn.calculate_for_each_entity(entity_ids)
        in_result = self.in_degree_info_loss_fn.calculate_for_each_entity(entity_ids)

        for entity_id in entity_ids:
            out_score = out_result[entity_id]
            in_score = in_result[entity_id]

            result[entity_id] = out_score * self.alpha_dm + (1 - self.alpha_dm) * in_score

        return result

    def call(self, entity_ids):
        result = self.calculate_for_each_entity(entity_ids)
        score = sum(result.values()) / len(entity_ids)

        return score