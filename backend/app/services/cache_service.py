import json
import logging
from typing import Optional, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class CacheService:
    """Simple caching service with Redis fallback to in-memory cache"""
    
    def __init__(self):
        self.available = False
        self.redis_client = None
        self.memory_cache = {}  # Fallback in-memory cache
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(settings.REDIS_URL)
                # Test connection
                self.redis_client.ping()
                self.available = True
                logger.info("Redis cache service initialized successfully")
            except Exception as e:
                logger.warning(f"Redis not available, using in-memory cache: {e}")
                self.available = False
        else:
            logger.warning("Redis not installed, using in-memory cache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.available and self.redis_client:
                # Try Redis first
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value.decode('utf-8'))
            else:
                # Fallback to memory cache
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or settings.CACHE_TTL
            serialized_value = json.dumps(value)
            
            if self.available and self.redis_client:
                # Use Redis
                self.redis_client.setex(key, ttl, serialized_value)
                return True
            else:
                # Fallback to memory cache (no TTL for simplicity)
                self.memory_cache[key] = value
                return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.available and self.redis_client:
                self.redis_client.delete(key)
            else:
                self.memory_cache.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            if self.available and self.redis_client:
                self.redis_client.flushdb()
            else:
                self.memory_cache.clear()
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

# Global cache instance
cache_service = CacheService()
