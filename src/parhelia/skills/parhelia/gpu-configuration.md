---
name: parhelia-gpu-configuration
description: GPU selection, optimization, and cost trade-offs
category: parhelia
keywords: [gpu, a10g, a100, h100, t4, cuda, ml, training, inference]
---

# GPU Configuration

**Scope**: Selecting and optimizing GPU resources for remote execution
**Lines**: ~280
**Last Updated**: 2026-01-21
**Format Version**: 1.0 (Atomic)

## When to Use This Skill

- Choosing the right GPU for your workload
- Understanding GPU cost vs performance trade-offs
- Optimizing GPU utilization
- Debugging GPU-related issues
- Configuring GPU memory and compute

## Core Concepts

### Available GPUs

| GPU | VRAM | Cost/hr | Best For |
|-----|------|---------|----------|
| **T4** | 16GB | ~$0.50 | Inference, light training |
| **A10G** | 24GB | ~$1.10 | Training, medium models |
| **A100** | 40/80GB | ~$2.50 | Large models, fast training |
| **H100** | 80GB | ~$4.50 | Largest models, fastest training |

### GPU Selection Matrix

| Task | Recommended GPU |
|------|-----------------|
| Small model inference (<7B params) | T4 |
| Medium model inference (7-13B) | A10G |
| Large model inference (>13B) | A100 |
| Fine-tuning small models | A10G |
| Fine-tuning large models | A100/H100 |
| Training from scratch | A100/H100 |
| Quick experiments | T4 |

## Patterns

### Pattern 1: Inference Task

**When**: Running predictions, not training

```bash
parhelia submit "Run inference on the test set" --gpu T4
```

T4 is cost-effective for inference workloads.

### Pattern 2: Training Task

**When**: Model training, fine-tuning

```bash
parhelia submit "Fine-tune the model for 10 epochs" --gpu A10G
```

A10G balances cost and capability for most training.

### Pattern 3: Large Model Work

**When**: Working with >13B parameter models

```bash
parhelia submit "Train the 70B model" --gpu A100
```

A100's 80GB VRAM handles large models.

### Pattern 4: Maximum Performance

**When**: Time-critical, budget allows

```bash
parhelia submit "Train as fast as possible" --gpu H100
```

H100 provides maximum throughput.

### Pattern 5: No GPU Needed

**When**: Non-ML tasks

```bash
parhelia submit "Run the test suite"
# No --gpu flag = CPU only
```

Save money when GPU isn't needed.

## VRAM Requirements Guide

| Model Size | Min VRAM | Recommended GPU |
|------------|----------|-----------------|
| <1B params | 4GB | T4 |
| 1-7B params | 16GB | T4/A10G |
| 7-13B params | 24GB | A10G |
| 13-30B params | 40GB | A100 |
| 30-70B params | 80GB | A100-80GB/H100 |
| >70B params | 80GB+ | H100 (or multi-GPU) |

## Cost Optimization

### Strategy 1: Start Small, Scale Up

```bash
# Try T4 first
parhelia submit "Training job" --gpu T4 --dry-run

# If OOM or too slow, upgrade
parhelia submit "Training job" --gpu A10G
```

### Strategy 2: Use CPU for Non-ML

```bash
# Don't waste GPU on tests
parhelia submit "Run pytest"
# No --gpu = ~$0.05/hr instead of $1+/hr
```

### Strategy 3: Batch Similar Work

```bash
# One GPU session for multiple tasks
parhelia submit "Train model A, then model B, then model C" --gpu A10G
# Better than 3 separate GPU sessions (saves cold start)
```

## Anti-Patterns

### Anti-Pattern 1: GPU for Everything

**Bad**: Using GPU for non-ML tasks
```bash
parhelia submit "Format the code" --gpu A100  # Waste!
```

**Good**: CPU for non-ML
```bash
parhelia submit "Format the code"  # No GPU
```

### Anti-Pattern 2: Oversized GPU

**Bad**: H100 for small inference
```bash
parhelia submit "Run small model prediction" --gpu H100  # Overkill
```

**Good**: Match GPU to workload
```bash
parhelia submit "Run small model prediction" --gpu T4
```

### Anti-Pattern 3: Undersized GPU

**Bad**: T4 for large model training
```bash
parhelia submit "Train 70B model" --gpu T4  # Will OOM
```

**Good**: Sufficient VRAM
```bash
parhelia submit "Train 70B model" --gpu A100
```

## Debugging GPU Issues

### Out of Memory (OOM)

```
CUDA out of memory
```

**Solutions**:
1. Upgrade to larger GPU
2. Reduce batch size
3. Enable gradient checkpointing
4. Use mixed precision (fp16/bf16)

### GPU Not Detected

```
No CUDA devices found
```

**Solutions**:
1. Verify `--gpu` flag was passed
2. Check Modal GPU availability in region
3. Try different GPU type

### Slow Performance

**Checklist**:
1. Is data loading the bottleneck? (CPU-bound)
2. Is batch size optimal?
3. Is the GPU actually being used? (check utilization)
4. Consider upgrading GPU tier

## Monitoring GPU Usage

When attached to session:

```bash
# Check GPU utilization
nvidia-smi

# Watch continuously
watch -n 1 nvidia-smi
```

Look for:
- GPU utilization % (should be high during training)
- Memory usage (should be near capacity)
- Temperature (throttling if too hot)

## Related Skills

- `parhelia/task-dispatch` - Task submission
- `parhelia/budget-management` - Cost control
- `parhelia/troubleshooting` - GPU debugging
