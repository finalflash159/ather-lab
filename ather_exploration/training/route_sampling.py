"""Bounded, stratified supervision drawn from current on-policy observations."""

import hashlib
from collections import Counter

import numpy as np


class BalancedRouteSamples:
    """Half disagreement, half ordinary; disjoint reservoirs and no duplicate fill.

    A seed/source contributes at most 16 candidates per rollout, limiting long
    loops. Reservoirs retain up to the total limit so either can fill a shortage.
    """

    def __init__(self, limit):
        self.limit = limit
        self.pools = {True: [], False: []}
        self.counts = Counter()
        self.seed_counts = Counter()
        self.seen = set()

    def offer(self, observation, labels, *, disagreement, ordinary, source):
        if not (disagreement or ordinary):
            return
        digest = hashlib.sha256(observation["memory"][[0, 1, 2, 3, 4, 7]].tobytes()).digest()
        if digest in self.seen or self.seed_counts[source] >= 16:
            return
        self.seen.add(digest)
        self.seed_counts[source] += 1
        category = bool(disagreement)
        self.counts[category] += 1
        pool = self.pools[category]
        slot = (
            len(pool) if len(pool) < self.limit else int(np.random.randint(self.counts[category]))
        )
        if slot < self.limit:
            item = ({k: v.copy() for k, v in observation.items()}, labels.copy())
            if slot == len(pool):
                pool.append(item)
            else:
                pool[slot] = item

    def select(self):
        wrong, ordinary = self.pools[True], self.pools[False]
        nwrong = min(len(wrong), (self.limit + 1) // 2)
        nordinary = min(len(ordinary), self.limit - nwrong)
        nwrong = min(len(wrong), self.limit - nordinary)
        result = []
        for pool, count in ((wrong, nwrong), (ordinary, nordinary)):
            if count:
                result.extend(pool[i] for i in np.random.choice(len(pool), count, replace=False))
        return result, nwrong
