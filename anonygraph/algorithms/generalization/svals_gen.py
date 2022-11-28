import logging

import anonygraph.info_loss.info as info

logger = logging.getLogger(__name__)

class SensitiveValuesGenerator:
    def __init__(self, fake_entity_manager):
        self.fake_entity_manager = fake_entity_manager

    def __call__(self, clusters, subgraph):
        logger.debug("clusters: {}".format(clusters))
        count = 0
        for cluster in clusters:
            signature = set()

            for entity_id in cluster:
                sval_id = subgraph.get_sensitive_value_id(entity_id)
                logger.debug("entity_id: {} - sval_id: {}".format(entity_id, sval_id))

                if sval_id is None:
                    sval_id = {self.fake_entity_manager.get_sensitive_value_id(entity_id)}
                    logger.debug("fake_entity_id: {} - sval_id: {}".format(entity_id, sval_id))

                signature.update(sval_id)


            logger.debug("cluster: {}".format(cluster))
            logger.debug("signature: {}".format(signature))
            logger.debug("sattr_id: {}".format(subgraph.sensitive_attr_id))
            for entity_id in cluster:
                for sval_id in signature:
                    subgraph.add_sensitive_value_id(entity_id, subgraph.sensitive_attr_id, sval_id)


            logger.debug("subgraph.entity2svals: {}".format(subgraph.entity2svals))

        # raise Exception()