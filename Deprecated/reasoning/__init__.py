"""
Advanced Reasoning Module for ResonantBrain SSM

This module implements two AGI-like capabilities:
1. Latent Thinking: Internal reasoning loops before output
2. Meta-Cognition: Pattern awareness and cross-domain transfer

Components:
    - thinking: ThinkingController and latent thinking loops
    - patterns: PatternMemoryBank for storing reasoning patterns
    - controller: MetaCognitiveController for pattern selection/application
"""

from .thinking import ThinkingController, LatentThinkingWrapper
from .patterns import ReasoningPattern, PatternMemoryBank, PatternDetector
from .controller import MetaCognitiveController

__all__ = [
    'ThinkingController',
    'LatentThinkingWrapper',
    'ReasoningPattern',
    'PatternMemoryBank',
    'PatternDetector',
    'MetaCognitiveController'
]
