"""
Knowledge Graph Loader - Wrapper for unified loader
This module provides backward compatibility by importing from the unified loader
"""

# Import everything from the unified loader for backward compatibility
from .kg_loader_unified import (
    KnowledgeGraphLoader,
    KnowledgeGraphBuilder,
    DataSourceType,
    DATA_SOURCES,
    CacheEntry,
    rate_limit
)

# For backward compatibility with old code that might use the old class name
KnowledgeGraphLoader = KnowledgeGraphLoader

__all__ = [
    'KnowledgeGraphLoader',
    'KnowledgeGraphBuilder',
    'DataSourceType',
    'DATA_SOURCES',
    'CacheEntry',
    'rate_limit'
]