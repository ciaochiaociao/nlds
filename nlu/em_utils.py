from typing import Tuple, List, Dict, Set
from nlu.data import EntityMention
from collections import defaultdict
from functools import reduce
import re


def same_type(fullids: List[str]) -> bool:
    """
    >>> same_type(['D0-S1-EM2', 'D9-S8-EM7'])
    True
    >>> same_type(['C0-S1-T2', 'D9-S8-EM7'])
    False
    """
    return bool(reduce(lambda id1, id2: id2 if re.split('[\d-]+', id1) == re.split('[\d-]+', id2) else False, fullids))


def id_cmp_key(fullid: str) -> Tuple[int]:
    return tuple([int(_) for _ in re.split('[A-Za-z-]+', fullid) if _])


def build_clusters_from_table(table, assert_same_type=True) -> List[Set[EntityMention]]:
    
    def _has_unending(fullid: str, table: Dict[str, EntityMention]) -> bool:
        """as long as there exists any mention that is unending (not at the end of an entity mention), return True; return False otherwise
        Args:
            - table: table should be ordered
        """
        # table should be ordered
        if fullid not in table or len(table[fullid]) == 0:  # no entity mentions at this token => ending
            return False
        else:
            # when >=1 entity mentions occupy this token
            ems = table[fullid]
            
            # as long as there exists any mention that is unending (not at the end of an entity mention), return True; return False otherwise
            for em in ems:
                if em.tokens[-1].fullid != fullid:
                    return True
            return False
    
    if assert_same_type:
        if len(table.keys()):
            assert same_type(table.keys())
    
    table = dict(sorted(table.items(), key=lambda t: id_cmp_key(t[0])))
    
    has_unendings: Dict[str, bool] = {fullid: _has_unending(fullid, table) for fullid in table.keys()}
    
    cluster = set()
    clusters: List[Set[EntityMention]] = []
    for fullid, unending in has_unendings.items():  
        ems = table[fullid]
        cluster.update(ems)
        if not unending:  # ending of a cluster
            if len(cluster):
                clusters.append(cluster)
            cluster = set()
    
    return clusters


def build_table(ems: List[EntityMention]) -> Dict[str, Set[EntityMention]]:

    _table = defaultdict(set)
    for em in ems:
        for token in em:
            _table[token.fullid].add(em)
    return _table


def build_clusters(ems):
    return build_clusters_from_table(build_table(ems))