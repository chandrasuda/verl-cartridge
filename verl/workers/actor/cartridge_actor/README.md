# Cartridge Actor for veRL

This package provides on-policy training of Cartridge caches within the veRL RL framework.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CartridgeActorWorker                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     CartridgePPOActor                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                    CacheAndModel                             │   │   │
│  │  │  ┌──────────────┐    ┌──────────────────────────────────┐  │   │   │
│  │  │  │TrainableCache│    │      Frozen HuggingFace Model    │  │   │   │
│  │  │  │  (trainable) │    │         (frozen)                 │  │   │   │
│  │  │  │              │    │                                  │  │   │   │
│  │  │  │• trainable_  │    │  • Base LLM (Llama/Qwen/etc)     │  │   │   │
│  │  │  │  keys        │◄───┤  • All params frozen             │  │   │   │
│  │  │  │• trainable_  │    │  • Gradient checkpointing        │  │   │   │
│  │  │  │  values      │    │    disabled                      │  │   │   │
│  │  │  └──────────────┘    └──────────────────────────────────┘  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              FSDP Wrapped (only cache sharded)               │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### CartridgePPOActor

Extends `BasePPOActor` to:
- Load/create a `TrainableCache`
- Wrap model with `CacheAndModel`
- Freeze base model parameters
- Only optimize cache parameters via `cache.parameters()`
- Use flex_attention for efficient attention with cache

### CartridgeActorWorker

Extends `ActorRolloutRefWorker` to:
- Detect `use_cartridge: true` in config
- Create `CartridgePPOActor` instead of standard actor
- Manage cartridge checkpointing
- Provide `save_cartridge()` method

## Configuration

```yaml
actor_rollout_ref:
  actor:
    # Enable cartridge training
    use_cartridge: true
    
    # Path to existing cartridge checkpoint (optional)
    cartridge_path: /path/to/cartridge.pt
    
    # Number of tokens to freeze at start of cache (default: 1)
    cartridge_num_frozen_tokens: 1
    
    # Standard veRL actor config
    ppo_epochs: 1
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 4
    grad_clip: 1.0
    
    optim:
      lr: 1e-4
      
  rollout:
    # Use Tokasaurus for rollout (Phase 1)
    name: tokasaurus
    mode: async
    custom:
      tokasaurus_url: "http://localhost:10210"
      cartridges:
        - id: "/path/to/cartridge.pt"
          source: "local"
```

## Training Flow

1. **Rollout Phase** (Tokasaurus)
   - Generate responses using `cartridge + prompt`
   - Returns: `input_ids`, `responses`, `log_probs` (from rollout)

2. **Actor Forward** (CartridgePPOActor)
   - Compute `log_probs` with `cache + input_ids`
   - Gradients flow into `TrainableCache` parameters
   - Base model remains frozen

3. **Policy Update** (CartridgePPOActor.update_policy)
   - Compute PPO/GRPO loss
   - Backward pass updates only cache parameters
   - Optimizer: `Adam(cache.parameters(), lr=1e-4)`

4. **Cartridge Sync** (Phase 3)
   - Save updated cartridge: `cache.save(path)`
   - Reload in Tokasaurus for next rollout

## Usage

```python
from verl.workers.actor.cartridge_actor import CartridgeActorWorker

# In your veRL trainer config
worker = CartridgeActorWorker(
    config={
        "actor": {
            "use_cartridge": True,
            "cartridge_path": "/tmp/cartridge.pt",
            "cartridge_num_frozen_tokens": 1,
            # ... other actor config
        },
        "rollout": {
            "name": "tokasaurus",
            # ... rollout config
        }
    },
    role="actor_rollout"
)
```

## Integration with Tokasaurus

The CartridgeActorWorker is designed to work with the Tokasaurus rollout (Phase 1):

1. Tokasaurus generates text using the cartridge
2. CartridgePPOActor computes gradients into the same cartridge
3. After update, save cartridge and signal Tokasaurus to reload

This creates a complete on-policy training loop for Cartridge caches.
