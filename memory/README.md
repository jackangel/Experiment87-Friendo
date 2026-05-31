# Saliency-Guided State Memory System

A novel post-training memory architecture for ResonantBrain SSM that stores and retrieves compressed SSM carry states instead of raw text.

## Quick Start

```python
from memory import MemoryConsolidation, MemoryRouter

# Initialize
consolidation = MemoryConsolidation(device='cuda', episodic_capacity=50)
router = MemoryRouter(consolidation, device='cuda')

# Store memory
store_conversation_memory(model, tokenizer, "Important text", consolidation, device)

# Generate with memory
generated_ids = generate_with_memory(
    model, context_ids, tokenizer, device, router,
    fusion_weight=0.3,
    max_memories_per_retrieval=3
)
```

## Modules

- **core.py**: SSMStateMemoryBank and MemoryEntry management
- **consolidation.py**: Episodic→Semantic memory consolidation
- **router.py**: Memory retrieval and state fusion

## Key Features

✓ **State-space memory**: Stores compressed SSM states (63 KB vs. 5 MB per memory)  
✓ **Saliency filtering**: Only stores important tokens (top 20%)  
✓ **Biological consolidation**: Episodic→Semantic with cognitive forgetting  
✓ **State fusion**: Blends memories directly into hidden states (no context waste)  
✓ **Scalable**: Handles 10,000+ memories efficiently

## Documentation

See [MEMORY_SYSTEM_GUIDE.md](../MEMORY_SYSTEM_GUIDE.md) for complete documentation.

## Architecture

```
Episodic Buffer (recent) → Semantic Memory (long-term)
         ↓                           ↓
    Consolidation               Retrieval
         ↓                           ↓
    Time decay              Similarity search
    Access tracking         State fusion
    Memory merging          Periodic injection
```

## Injection Modes

1. **state_fusion** (recommended): Blend carry states directly
2. **kv_injection**: Prepend high-saliency KV cache entries  
3. **context_prepend**: Traditional RAG (text concatenation)

## Performance

- **Memory per entry**: ~63 KB (compressed states + metadata)
- **Retrieval speed**: ~2ms for 1000 memories
- **Generation overhead**: ~5-10% (with periodic retrieval every 50 tokens)

## Example

```python
# Initialize memory system
consolidation, router = initialize_memory_system(device='cuda')

# Store important information
store_conversation_memory(
    model, tokenizer, 
    "User prefers concise technical explanations",
    consolidation, device,
    metadata={'type': 'user_profile'}
)

# Generate with memory augmentation
response = generate_with_memory(
    model, prompt_tokens, tokenizer, device, router,
    memory_retrieval_interval=50,  # Retrieve every 50 tokens
    fusion_weight=0.3              # 30% memory, 70% current
)

# Save memories for next session
consolidation.save_to_disk('memories.pt')
```

## Requirements

- PyTorch >= 2.0
- ResonantBrain SSM model
- CUDA-capable GPU (recommended)

## License

Same as parent ResonantBrain project.
