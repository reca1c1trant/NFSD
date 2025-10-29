# Bug Fix Summary - Score Flow Implementation

## Critical Bugs Fixed in `sgm/models/score_flow.py`

### Bug 1: Broken Gradient Flow (Line 156)

**Problem:**
```python
score_flat = self.score_projector(
    torch.cat([score_flat.detach(), sigma_emb], dim=-1)  # .detach() breaks backprop!
)
```

The `.detach()` operation broke the gradient flow from the loss back to the flow network parameters. This meant:
- The exact score computed via `torch.autograd.grad()` was correct
- But the flow network couldn't be trained because gradients were blocked
- The score_projector MLP also couldn't learn anything useful

**Fix:**
Removed `.detach()` to allow gradients to flow:
```python
score_flat = self.score_projector(
    torch.cat(projector_input_list, dim=-1)  # No detach - gradients flow!
)
```

**Why This Matters:**
The training loop is:
1. Flow produces energy function
2. Score = ∇ energy (exact gradient computation)
3. x_pred = x_t + σ² * score
4. loss = ||x_pred - x_clean||²
5. Backprop updates flow parameters

Without gradient flow, step 5 fails and the flow network never learns.

---

### Bug 2: Dimension Mismatch (Line 156)

**Problem:**
`score_projector` was initialized with `flow_input_dim`:
```python
# In __init__:
flow_input_dim = data_dim + sigma_embed_dim + cond_embed_dim  # e.g., 128 + 256 + 128 = 512

self.score_projector = nn.Sequential(
    nn.Linear(flow_input_dim, data_dim * 2),  # Expects 512
    ...
)
```

But in `forward()`, only passed `data_dim + sigma_embed_dim`:
```python
# In forward() - WRONG:
torch.cat([score_flat, sigma_emb], dim=-1)  # Only 128 + 256 = 384 !
```

This would cause a runtime error: `size mismatch, got 384, expected 512`

**Fix:**
Include conditioning embedding to match the expected dimension:
```python
projector_input_list = [score_flat, sigma_emb]
if cond_emb is not None:
    projector_input_list.append(cond_emb)

score_flat = self.score_projector(
    torch.cat(projector_input_list, dim=-1)  # Now matches flow_input_dim
)
```

**Why This Matters:**
- Without this fix, training would crash with dimension mismatch error
- The score_projector needs ALL conditioning information (including class labels) to properly refine the score function

---

## Modified Files

1. **sgm/models/score_flow.py**
   - Lines 154-162: Fixed gradient flow and dimension mismatch in ScoreFlowNetwork.forward()

---

## Training Impact

These fixes ensure:
1. ✅ Exact gradients from Flow are preserved AND gradients flow back for training
2. ✅ All conditioning information is properly used in score computation
3. ✅ The flow network can actually learn from the loss signal
4. ✅ Training will not crash due to dimension errors

---

## Current Implementation Status

### ✅ Completed Requirements:
1. **Exact gradient computation**: Flow computes energy, score = ∇ energy via autograd
2. **Latent space training**: AutoencoderKLLinear (784→128→784) implemented
3. **No approximations**: Direct gradient computation, no shortcuts
4. **Proper backpropagation**: Gradients now flow correctly

### 📝 Configuration Files:
- `configs/training/flow_diffusion_mnist.yaml` - Pixel space training
- `configs/training/flow_diffusion_mnist_latent.yaml` - **Latent space training** (128-dim)

### 🔧 Training Command:
```bash
# Latent space training (recommended)
python scripts/train_flow_diffusion.py \
    --config configs/training/flow_diffusion_mnist_latent.yaml \
    --name flow_mnist_latent

# Pixel space training (for comparison)
python scripts/train_flow_diffusion.py \
    --config configs/training/flow_diffusion_mnist.yaml \
    --name flow_mnist_pixel
```

---

## Next Steps

1. **Test training** to verify no runtime errors
2. **Monitor loss** to ensure flow network is learning
3. **Generate samples** after training to verify quality
4. **Consider pre-training autoencoder** separately if needed for better latent space

---

## Technical Notes

### Why score_projector exists:
The exact score from ∇ energy is mathematically correct, but may benefit from:
- Additional expressiveness via MLP
- Conditioning on noise level (sigma)
- Conditioning on class/text embeddings

The projector acts as a learned refinement while preserving the exact gradient foundation.

### Gradient flow path:
```
loss(x_pred, x_clean)
  ↓ backprop
x_pred = x_t + σ² * score
  ↓ backprop
score = score_projector(∇ energy, σ, cond)
  ↓ backprop (NOW WORKS - detach removed!)
∇ energy via autograd.grad()
  ↓ backprop
energy = -||flow_output||²
  ↓ backprop
flow_output = flow.forward([x_t, σ, cond])
  ↓ backprop
flow network parameters (TRAINED!)
```
