"""Step 6: cache recommendation lists in Redis (key rec:<variant>:<user>:<k>, JSON value, TTL).

If Redis is not reachable the cache falls back to an in-process dict, so the API and the
offline A/B test still run on a laptop without Redis. Start Redis with:  docker compose up -d redis
"""
import json
import time

import redis

from src.common.config import REDIS_URL


class RecCache:
    def __init__(self, url=REDIS_URL, ttl_s=3600):
        self.ttl = ttl_s
        self.hits = self.misses = 0
        try:
            self.r = redis.Redis.from_url(url, socket_connect_timeout=0.5, socket_timeout=0.5)
            self.r.ping()
            self.backend = "redis"
        except redis.RedisError:
            self.r, self.local = None, {}
            self.backend = "memory"
        print(f"[cache] backend={self.backend}")

    @staticmethod
    def key(variant, user_id, k):
        return f"rec:{variant}:{user_id}:{k}"

    def get(self, key):
        if self.r is not None:
            raw = self.r.get(key)
            value = json.loads(raw) if raw else None
        else:
            value, expires = self.local.get(key, (None, 0))
            value = value if expires > time.time() else None
        self.hits += value is not None
        self.misses += value is None
        return value

    def set(self, key, value):
        if self.r is not None:
            self.r.set(key, json.dumps(value), ex=self.ttl)
        else:
            self.local[key] = (value, time.time() + self.ttl)

    def clear(self):
        if self.r is not None:
            for k in self.r.scan_iter("rec:*"):
                self.r.delete(k)
        else:
            self.local.clear()
