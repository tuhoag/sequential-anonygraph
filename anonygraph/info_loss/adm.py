import logging

logger = logging.getLogger(__name__)

from .base_info_loss import BaseInfoLossMetric
from .am import AttributeInfoLoss
from .dm import OutDegreeInfoLoss, InDegreeInfoLoss, OutInDegreeInfoLoss


class AttributeOutInDegreeInfoLoss(BaseInfoLossMetric):
    def __init__(self, subgraph, args):
        super().__init__(subgraph, args)

        self.attribute_info_loss_fn = AttributeInfoLoss(subgraph, args)
        self.out_in_degree_info_loss_fn = OutInDegreeInfoLoss(subgraph, args)

        self.alpha_adm = args['alpha_adm']

    def calculate_for_each_entity(self, entity_ids):
        am_result = self.attribute_info_loss_fn.calculate_for_each_entity(entity_ids)
        dm_result = self.out_in_degree_info_loss_fn.calculate_for_each_entity(entity_ids)

        result = {}
        for entity_id in entity_ids:
            am_score = am_result[entity_id]
            dm_score = dm_result[entity_id]

            result[entity_id] = am_score * self.alpha_adm + (1 - self.alpha_adm) * dm_score

        return result

    def call(self, entity_ids):
        result = self.calculate_for_each_entity(entity_ids)
        score = sum(result.values()) / len(entity_ids)

        return score