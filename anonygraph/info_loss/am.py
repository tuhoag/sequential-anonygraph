import logging

logger = logging.getLogger(__name__)

from .base_info_loss import BaseInfoLossMetric
from .info import get_generalized_attribute_info

class AttributeInfoLoss(BaseInfoLossMetric):
    def calculate_for_each_entity(self, entity_ids):
        result = {}
        for entity_id in entity_ids:
            result[entity_id] = 0.0

        if len(entity_ids) == 0:
            return result

        union_info, entities_info = get_generalized_attribute_info(self.subgraph, entity_ids)
        num_attributes = self.subgraph.num_attribute_relations
        num_fake_or_removed_entities = 0

        logger.debug("union: {} - entities ({}): {} ".format(union_info, entity_ids, entities_info))
        if len(union_info) != 0:
            num_real_entities = 0
            for entity_id, entity_info in entities_info.items():
                if self.subgraph.is_entity_id(entity_id):
                    num_real_entities += 1
                    entity_score = 0.0

                    for relation_id, union_value_ids in union_info.items():
                        max_num_value_ids = self.subgraph.get_num_domain_value_ids_from_relation_id(
                            relation_id)
                        num_value_ids = len(union_value_ids)
                        num_entity_value_ids = len(entity_info.get(relation_id, set()))
                        if max_num_value_ids - num_entity_value_ids + 1 == 0:
                            raise Exception("relation: {} union: {} domain: {}".format(relation_id, union_value_ids, self.subgraph.get_domain_value_ids(relation_id)))
                            raise Exception("max: {} - num: {}".format(max_num_value_ids, num_entity_value_ids))
                        current_score = (num_value_ids - num_entity_value_ids) / (max_num_value_ids - num_entity_value_ids + 1)

                        entity_score += current_score

                        logger.debug("relation: {} union: {} - info: {} - max: {} - score: {}".format(relation_id, num_value_ids, num_entity_value_ids, max_num_value_ids, current_score))

                    entity_score = entity_score / num_attributes
                else:
                    entity_score = 1
                    # score += 1
                    num_fake_or_removed_entities += 1

                result[entity_id] = entity_score

        return result

    def call(self, entity_ids):
        result = self.calculate_for_each_entity(entity_ids)
        score = sum(result.values()) / len(entity_ids)

        return score