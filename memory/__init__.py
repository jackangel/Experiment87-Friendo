"""
Saliency-Guided State Memory System for ResonantBrain SSM

This memory system leverages the architecture's unique features:
- SSM carry states for compressed sequence representations
- Saliency tracking for importance-based storage
- Cognitive forgetting-inspired consolidation

Architecture:
    Episodic Memory (recent, short-term) → Semantic Memory (long-term, consolidated)
    
Components:
    - core: SSMStateMemoryBank and memory entry management
    - consolidation: Episodic→Semantic transition with forgetting
    - router: Retrieval and state fusion into inference
"""

from .core import SSMStateMemoryBank, MemoryEntry
from .consolidation import MemoryConsolidation
from .router import MemoryRouter

__all__ = [
    'SSMStateMemoryBank',
    'MemoryEntry',
    'MemoryConsolidation',
    'MemoryRouter'
]
