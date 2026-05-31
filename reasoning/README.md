# AGI Reasoning Systems

Advanced reasoning capabilities for ResonantBrain SSM.

## Quick Start

```python
# Initialize latent thinking
thinking_model = initialize_latent_thinking(model, tokenizer)

# Generate with internal reasoning
generated_ids, info = thinking_model.generate_with_thinking(
    context_ids, device,
    max_new_tokens=256,
    thinking_threshold=0.7
)

print(f"Thinking steps: {info['total_thinking_steps']}")
```

## Modules

### thinking.py
- **ThinkingController**: Confidence assessment and depth prediction
- **LatentThinkingWrapper**: Wraps model with thinking capability

### patterns.py
- **ReasoningPattern**: Reasoning pattern data structure
- **PatternMemoryBank**: Store and retrieve patterns
- **PatternDetector**: Detect active reasoning patterns

### controller.py
- **MetaCognitiveController**: High-level pattern management
- **generate_with_metacognition()**: Generate with pattern transfer

## Features

✓ **Latent Thinking**: Internal reasoning loops (1-5 steps)  
✓ **Adaptive Depth**: Learns when and how much to think  
✓ **Pattern Memory**: Stores abstract reasoning strategies  
✓ **Cross-Domain Transfer**: Apply patterns across domains  
✓ **Success Tracking**: Monitor pattern effectiveness  
✓ **Confidence-Based**: Only thinks when uncertain  

## Usage Modes

1. **Thinking Only** (Mode 5): Internal reasoning loops
2. **Meta-Cognition Only** (Mode 6): Pattern transfer
3. **Full AGI Stack** (Mode 7): All capabilities combined

## Key Concepts

**Latent Thinking**: Like Chain-of-Thought but internal (no token output)
- Complex problem → Think 3-5 steps → Output answer
- Simple problem → Output immediately

**Meta-Cognition**: Explicit pattern storage and transfer
- Learn modus ponens in logic → Apply to code reasoning
- Learn chain rule in math → Apply to language composition

## Performance

- **Thinking overhead**: ~10-30% slower (adaptive, only when needed)
- **Pattern retrieval**: ~2ms for 500 patterns
- **Memory per pattern**: ~50 KB (compressed carry states)
- **Cross-domain success**: 60-80% (empirical, task-dependent)

## Documentation

See [AGI_SYSTEMS_GUIDE.md](../AGI_SYSTEMS_GUIDE.md) for complete guide.

## Requirements

- PyTorch >= 2.0
- ResonantBrain SSM model
- CUDA-capable GPU (recommended)

## License

Same as parent ResonantBrain project.
