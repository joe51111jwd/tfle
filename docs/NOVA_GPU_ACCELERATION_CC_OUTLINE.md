# NOVA Complete Build — Claude Code Implementation Outline

> **Purpose**: Give any CC session everything needed to (1) make TFLE fully GPU-accelerated and (2) build all missing components from the original NOVA vision. Read NOVA_CC_BRIEFING.md for full project context. Read the three theory docs (compression-driven-learning.md, sleep-wake-training.md, trit-flip-local-evolution.md) for the science behind each component.

---

## SITUATION SUMMARY

### What Exists (in `/tfle/`)
- TFLE algorithm with task-loss fitness (got 23.54% on MNIST, up from 10.31% random chance)
- Ternary layers, MLP model, STE baseline (89.28%), config, training loop, monitoring
- Conv layers and conv model for CIFAR-10 (untested)
- Transformer (BROKEN — uses `torch.randn()` instead of real activations at line ~223)
- fitness.py with 3 fitness functions (implemented but disconnected from training)

### What's Missing (MUST BUILD)
1. **GPU support** — zero `.to(device)` calls anywhere, everything runs on CPU
2. **Batched parallel evolution** — evaluates 1 flip proposal at a time, GPU sits idle
3. **CDLL (Compression-Driven Layer Learning)** — the local fitness function, completely unimplemented
4. **SWT (Sleep-Wake Training)** — the continual learning system, completely unimplemented
5. **Mono-Forward local classifier heads** — per-layer error signals for layer-parallel training
6. **Full NOVA model architecture** — hybrid Transformer-Mamba, BitLinear, MoLoRA (none built)
7. **All 5 intelligence strategies** — execution verify, consensus, curiosity, tool orchestration, adversarial review
8. **Training pipeline** — pretraining, distillation, GRPO
9. **Agent loop** — task parser, planner, environment, memory
10. **Benchmarks** — evaluation suite

### What This Session Should Build
The CC session should focus on making TFLE work on GPU with batched evolution, then build the missing CDLL and SWT components. The full model architecture and inference strategies are later priorities.

---

## PART 1: GPU ACCELERATION

### Why TFLE is Currently CPU-Bound

The entire codebase has zero device management. All tensors are created on CPU by default. Even if moved to GPU, the algorithm is sequential: for each layer → propose 1 flip set → 2 full-model forward passes → accept/reject. For MNIST this means microsecond operations with massive Python overhead between them.

### The Fix: Batched Within-Layer Evolution

**Core insight**: Instead of evaluating 1 flip proposal per layer, evaluate **K proposals simultaneously** as a single batched GPU operation. This is how Evolution Strategies work.

**Why this is safe**: You're testing K different flip combinations for the same layer and picking the best. Same fitness function, same accept/reject logic — just smarter search. Like trying 32 keys in a lock at once.

**Why cross-layer batching with task-loss is NOT safe**: Task-loss requires full-model forward passes. Layer 0's accepted changes affect layer 1's fitness evaluation. Training all layers in parallel would mean evaluating against stale weights. This would break learning. **Keep layers sequential when using task-loss.**

**Why cross-layer batching WITH local fitness IS safe**: CDLL and Mono-Forward give each layer an independent fitness signal. Layers don't need to know about each other's weights. All layers can train in parallel. **This is why building CDLL matters for GPU performance, not just theory.**

### Step 1.1: Device Management

**Files to modify**: `config.py`, `layers.py`, `model.py`, `training.py`, `experiments/phase1_mnist.py`

Add to `TFLEConfig`:
```python
device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
num_parallel_proposals: int = 32  # K — proposals evaluated in parallel
proposal_diversity: float = 0.5   # fraction of candidates flipped per proposal
pin_memory: bool = True
num_workers: int = 4
```

Add utility:
```python
def resolve_device(config) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.device)
```

**What to change in every file:**
- `TFLELayer.__init__`: Accept `device` param. Create `self.weights`, `self.traces`, `self.success_traces`, `self.error_traces` all on `device`
- `TFLELayer.forward()`: Weights `.float()` already returns same device if input is on device — but verify
- `TFLELayer._select_candidates()`: `torch.randint`, `torch.randperm`, `torch.topk` etc. need `device=` kwarg
- `TFLELayer._propose_flips()`: All tensor creation needs `device=`
- `TFLEModel.__init__`: Resolve device, pass to all layers, store `self.device`
- `TFLETrainer.train()`: `batch_x = batch_x.to(self.model.device)`, `batch_y = batch_y.to(self.model.device)`
- `TFLETrainer._evaluate()`: Same device moves
- `experiments/phase1_mnist.py`: `DataLoader(pin_memory=True, num_workers=4)`, print device info

### Step 1.2: Vectorized Flip Proposal Generation

Replace the Python loop in `_propose_flips()` with vectorized GPU ops:

```python
def generate_k_proposals(weights, candidates, K, device):
    """Generate K different flip proposals, fully vectorized on GPU.

    weights: [in_feat, out_feat] current ternary weights
    candidates: [n_candidates] flat indices
    K: number of proposals

    Returns: [K, in_feat, out_feat] tensor of proposed weight matrices
    """
    flat_w = weights.flatten()  # [n_weights]
    proposals = flat_w.unsqueeze(0).expand(K, -1).clone()  # [K, n_weights]

    n_cand = candidates.shape[0]
    # Each proposal flips a random ~50% subset of candidates (diversity)
    flip_masks = torch.rand(K, n_cand, device=device) < 0.5

    current_vals = proposals[:, candidates]  # [K, n_cand]

    # Ternary flip: add random offset (1 or 2) mod 3, then shift to {-1,0,1}
    random_offsets = torch.randint(1, 3, (K, n_cand), device=device)
    new_vals = (current_vals + 1 + random_offsets) % 3 - 1  # maps correctly

    proposals[:, candidates] = torch.where(flip_masks, new_vals, current_vals)
    return proposals.reshape(K, weights.shape[0], weights.shape[1])
```

### Step 1.3: Batched Forward Pass Evaluation

The key GPU optimization. For task-loss fitness, instead of K separate forward passes:

```python
def batched_task_loss_eval(model, layer_idx, proposals_K, x, labels):
    """Evaluate K flip proposals for one layer using cached prefix/suffix.

    CRITICAL OPTIMIZATION: Only ONE layer changes between proposals.
    Cache everything before and after that layer.

    1. Prefix (layers 0..layer_idx-1): computed ONCE → intermediate activations
    2. Varying layer: K batched matmuls (the parallel part)
    3. Suffix (layers layer_idx+1..end): batched over K stacked outputs

    This turns 2K forward passes into ~3 passes of compute.
    """
    K = proposals_K.shape[0]

    with torch.no_grad():
        # 1. CACHE PREFIX — one forward pass through layers before this one
        h = x
        for i in range(layer_idx):
            h = model.layers[i].forward(h)
            if i < len(model.layers) - 1:
                h = F.relu(h)
        # h: [batch, in_features_for_this_layer]

        # 2. BATCHED VARYING LAYER — K different weight matrices
        h_expanded = h.unsqueeze(0).expand(K, -1, -1)  # [K, batch, in_feat]
        w_float = proposals_K.float()                    # [K, in_feat, out_feat]
        varied = torch.bmm(h_expanded, w_float)          # [K, batch, out_feat]

        if layer_idx < len(model.layers) - 1:
            varied = F.relu(varied)

        # 3. BATCHED SUFFIX — stack K outputs, run through remaining layers
        K_val, B, F_out = varied.shape
        h_flat = varied.reshape(K_val * B, F_out)
        for i in range(layer_idx + 1, len(model.layers)):
            h_flat = model.layers[i].forward(h_flat)
            if i < len(model.layers) - 1:
                h_flat = F.relu(h_flat)

        # 4. COMPUTE ALL K LOSSES AT ONCE
        logits = h_flat.reshape(K_val, B, -1)
        labels_exp = labels.unsqueeze(0).expand(K_val, -1)
        losses = F.cross_entropy(
            logits.reshape(K_val * B, -1),
            labels_exp.reshape(K_val * B),
            reduction='none'
        ).reshape(K_val, B).mean(dim=1)  # [K]

    return losses  # lower = better
```

### Step 1.4: Updated Training Step

In `model.py`, new `_train_step_task_loss_batched()`:

```python
def _train_step_task_loss_batched(self, x, temperature, labels):
    all_metrics = []
    K = self.config.num_parallel_proposals

    for layer_idx, layer in enumerate(self.layers):
        loss_before = self._compute_task_loss(x, labels)

        # Generate K proposals
        layer_temp = self.config.get_temperature_for_layer(temperature, layer_idx)
        combined_traces = layer._get_combined_traces()
        candidates = layer._select_candidates(combined_traces)
        proposals_K = generate_k_proposals(
            layer.weights, candidates, K, device=self.device
        )

        # Also include current weights as proposal 0 (baseline)
        current_flat = layer.weights.flatten().unsqueeze(0)
        all_proposals = torch.cat([
            current_flat.reshape(1, *layer.weights.shape),
            proposals_K
        ], dim=0)  # [K+1, in, out]

        # Evaluate all K+1 proposals at once
        losses = batched_task_loss_eval(self, layer_idx, all_proposals, x, labels)

        # Proposal 0 is current weights (baseline)
        loss_current = losses[0].item()
        losses_proposals = losses[1:]
        best_k = losses_proposals.argmin()
        best_loss = losses_proposals[best_k].item()

        fitness_delta = loss_current - best_loss

        # Accept/reject
        if fitness_delta > 0:
            accepted = True
        elif layer_temp <= 0:
            accepted = False
        else:
            prob = math.exp(min(fitness_delta / max(layer_temp, 1e-8), 0))
            accepted = torch.rand(1).item() < prob

        if accepted:
            layer.apply_proposal(proposals_K[best_k])
        else:
            layer.reject_proposal(candidates)

        # Trace update
        with torch.no_grad():
            preds = self.predict(x)
            batch_acc = (preds == labels).float().mean().item()
        layer.update_traces_from_task(x, batch_acc < 0.9)

        all_metrics.append({
            "accepted": accepted,
            "fitness_delta": fitness_delta,
            "loss_before": loss_current,
            "loss_after": best_loss if accepted else loss_current,
            "n_candidates": len(candidates),
            "temperature": layer_temp,
            "batch_accuracy": batch_acc,
            "proposals_evaluated": K,
        })

    return all_metrics
```

### Step 1.5: Recommended K Values

| Model Scale | K (proposals) | Why |
|-------------|---------------|-----|
| MNIST MLP (tiny) | 64-256 | Model is small, GPU can handle many proposals |
| CIFAR-10 CNN | 32-64 | Medium model, good GPU saturation |
| 2.4B full model | 8-16 | Each forward pass is large enough to saturate GPU on its own |

### Step 1.6: Expected Performance After Phase 1

| Setup | Time (100K steps MNIST) | GPU Util |
|-------|------------------------|----------|
| Current (CPU, K=1) | 4-8 hours | 0% |
| Phase 1 (GPU, K=32) | 10-30 min | 60-80% |
| Phase 1 (GPU, K=128) | 5-15 min | 80-90% |

Accuracy should IMPROVE because K>1 means better search per step (testing 32 candidates instead of 1).

---

## PART 2: CDLL — COMPRESSION-DRIVEN LAYER LEARNING

**Status**: Completely unimplemented. Config has `FitnessType.COMPRESSION` but no code.

**What it is**: Each layer's fitness = how well it compresses its input while preserving task-relevant information. Based on the Information Bottleneck principle.

**Why it matters for GPU**: CDLL is a LOCAL fitness function — it only needs the layer's input and output. No full-model forward pass. This means:
- Flip evaluation is O(layer) not O(model)
- All layers can train in parallel (their fitness is independent)
- This is what makes TFLE feasible at 2.4B scale

### Create: `tfle/cdll.py`

```python
class CDLLFitness:
    """Compression-Driven Layer Learning fitness function.

    L_layer = alpha * H(output) - beta * I(output; input_features)

    Where:
      H(output) = entropy of activation patterns (lower = more compressed)
      I(output; input_features) = mutual information (higher = more preserved)

    For ternary networks, entropy is EXACT (discrete activations).
    """

    def __init__(self, alpha=1.0, beta=1.0, layer_depth=0, total_layers=1,
                 mi_method='variance', device='cpu'):
        # Deeper layers compress MORE
        self.alpha = alpha * (1 + 0.5 * layer_depth / max(total_layers - 1, 1))
        self.beta = beta
        self.mi_method = mi_method  # 'variance', 'reconstruction', 'infonce'
        self.device = device

        # For reconstruction proxy
        self.decoder = None  # tiny decoder, built on first call

    def compute_fitness(self, layer_input, layer_output):
        """Compute CDLL fitness. Higher = better.

        Returns: -alpha * entropy + beta * mutual_info
        """
        entropy = self._estimate_entropy(layer_output)
        mi = self._estimate_mutual_info(layer_input, layer_output)
        return -self.alpha * entropy + self.beta * mi

    def _estimate_entropy(self, activations):
        """Estimate entropy of activation patterns across the batch.

        For post-ReLU activations: bin into discrete buckets.
        For ternary (post-quantization): exact computation.
        """
        # Binned entropy estimation for continuous activations
        # Discretize each neuron's output into bins
        B, D = activations.shape
        n_bins = 32  # reasonable for mini-batch estimation

        entropy_sum = 0.0
        for d in range(min(D, 256)):  # sample neurons if too many
            col = activations[:, d]
            if col.max() == col.min():
                continue  # zero entropy
            # Histogram-based entropy
            hist = torch.histc(col, bins=n_bins)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            entropy_sum += -(probs * probs.log2()).sum().item()

        return entropy_sum / min(D, 256)

    def _estimate_mutual_info(self, input_act, output_act):
        """Estimate mutual information between layer input and output."""
        if self.mi_method == 'variance':
            return self._mi_variance(input_act, output_act)
        elif self.mi_method == 'reconstruction':
            return self._mi_reconstruction(input_act, output_act)
        else:
            return self._mi_variance(input_act, output_act)

    def _mi_variance(self, input_act, output_act):
        """Variance-based MI proxy.

        High output variance along input-correlated dimensions = high MI.
        """
        # Cross-covariance between input and output
        x_centered = input_act - input_act.mean(dim=0)
        y_centered = output_act - output_act.mean(dim=0)

        # Frobenius norm of cross-covariance matrix (sampled)
        # This is proportional to MI for Gaussian variables
        n = x_centered.shape[0]
        # Sample dimensions if too large
        x_dim = min(x_centered.shape[1], 256)
        y_dim = min(y_centered.shape[1], 256)
        cross_cov = (x_centered[:, :x_dim].T @ y_centered[:, :y_dim]) / n
        return cross_cov.norm().item()

    def _mi_reconstruction(self, input_act, output_act):
        """Reconstruction proxy: can a tiny decoder recover input from output?

        Negative reconstruction error = proxy for mutual information.
        """
        if self.decoder is None:
            in_dim = input_act.shape[1]
            out_dim = output_act.shape[1]
            self.decoder = nn.Sequential(
                nn.Linear(out_dim, min(out_dim * 2, in_dim)),
                nn.ReLU(),
                nn.Linear(min(out_dim * 2, in_dim), in_dim)
            ).to(self.device)
            self.dec_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)

        # Quick decoder update (1 step)
        self.dec_optimizer.zero_grad()
        reconstructed = self.decoder(output_act.detach())
        loss = F.mse_loss(reconstructed, input_act.detach())
        loss.backward()
        self.dec_optimizer.step()

        return -loss.item()  # higher = better preservation
```

### Wire Into Training

Add `FitnessType.CDLL` handling in `model.py`:
```python
def _train_step_cdll(self, x, temperature, labels=None):
    """CDLL fitness: each layer trains independently using compression metric.

    LAYERS CAN RUN IN PARALLEL HERE — each layer's fitness depends only
    on its own input and output, not on other layers.
    """
    all_metrics = []
    K = self.config.num_parallel_proposals

    # Compute each layer's input by forward-passing through preceding layers
    layer_inputs = [x]
    h = x
    for i, layer in enumerate(self.layers):
        layer_inputs.append(F.relu(layer.forward(h)) if i < len(self.layers) - 1 else layer.forward(h))
        h = layer_inputs[-1]

    # Now train each layer independently
    for layer_idx, layer in enumerate(self.layers):
        layer_in = layer_inputs[layer_idx]
        cdll = self.cdll_fitness[layer_idx]

        # Current fitness
        current_out = layer.forward(layer_in)
        if layer_idx < len(self.layers) - 1:
            current_out = F.relu(current_out)
        fitness_before = cdll.compute_fitness(layer_in, current_out)

        # Generate and evaluate K proposals
        candidates = layer._select_candidates(layer._get_combined_traces())
        proposals_K = generate_k_proposals(layer.weights, candidates, K, self.device)

        # Batched evaluation — just this layer, no full model pass
        layer_in_exp = layer_in.unsqueeze(0).expand(K, -1, -1)
        w_float = proposals_K.float()
        outputs_K = torch.bmm(layer_in_exp, w_float)  # [K, batch, out]
        if layer_idx < len(self.layers) - 1:
            outputs_K = F.relu(outputs_K)

        # Compute fitness for each proposal
        fitnesses = []
        for k in range(K):
            f = cdll.compute_fitness(layer_in, outputs_K[k])
            fitnesses.append(f)
        fitnesses = torch.tensor(fitnesses, device=self.device)

        best_k = fitnesses.argmax()
        fitness_delta = fitnesses[best_k].item() - fitness_before

        # Accept/reject with Boltzmann
        layer_temp = self.config.get_temperature_for_layer(temperature, layer_idx)
        if fitness_delta > 0:
            accepted = True
        elif layer_temp <= 0:
            accepted = False
        else:
            prob = math.exp(min(fitness_delta / max(layer_temp, 1e-8), 0))
            accepted = torch.rand(1).item() < prob

        if accepted:
            layer.apply_proposal(proposals_K[best_k])

        all_metrics.append({...})

    return all_metrics
```

### Refresh Layer Inputs

Since layers train with local fitness on cached inputs, inputs can become stale. **Every N steps (5-10), do a full forward pass to refresh `layer_inputs`.** This prevents drift.

---

## PART 3: MONO-FORWARD LOCAL CLASSIFIER HEADS

**Status**: Not implemented. Referenced as "Option B" in NOVA_CC_BRIEFING.md Section 6.

**What it is**: Each layer gets a tiny classifier that predicts labels from that layer's activations. Fitness = does the classification improve after flips?

**Why it matters**: Provides label-aware local signal (stronger than CDLL alone). Research shows Mono-Forward EXCEEDS backprop on CIFAR-10/100 (see briefing Section 5). 34% faster, 41% less energy.

### Create: `tfle/local_heads.py`

```python
class LocalClassifierHead(nn.Module):
    """Tiny classifier attached to a TFLE layer for local error signals.

    This trains with standard backprop (it's tiny, ~50K params).
    Only the main ternary weights train via TFLE.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=128, device='cpu'):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)

    def compute_fitness(self, activations, labels):
        """Local fitness: cross-entropy loss of predicting labels from activations."""
        with torch.no_grad():
            logits = self.classifier(activations)
            loss = F.cross_entropy(logits, labels)
        return -loss.item()  # higher = better

    def compute_accuracy(self, activations, labels):
        """For monitoring."""
        with torch.no_grad():
            preds = self.classifier(activations).argmax(dim=-1)
            return (preds == labels).float().mean().item()

    def update(self, activations, labels):
        """Train the local classifier (cheap, standard backprop)."""
        self.optimizer.zero_grad()
        logits = self.classifier(activations.detach())
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

### Hybrid: CDLL + Mono-Forward

Use both for strongest signal:
```python
combined_fitness = mono_forward_fitness + lambda_cdll * cdll_fitness
```
The classifier ensures task-relevance. CDLL ensures compression quality.

---

## PART 4: SWT — SLEEP-WAKE TRAINING

**Status**: Completely unimplemented. This is one of the three core pillars.

**What it is**: A two-phase continual learning system inspired by how the brain separates fast learning (wake) from slow consolidation (sleep).

### Create: `tfle/swt.py`

```python
class SleepWakeTrainer:
    """Sleep-Wake Training for continual learning.

    Wake Phase: Fast, local TFLE updates from streaming data
    Sleep Phase: Consolidation via replay, EWC, and micro-critics
    """

    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device

        # Replay buffer — stores (input, label, surprise_score) tuples
        self.replay_buffer = ReplayBuffer(
            max_size=config.swt.replay_buffer_size,  # 10000
            priority='surprise'
        )

        # EWC — Fisher information for weight protection
        self.fisher_info = {}  # layer_idx -> fisher matrix
        self.ewc_lambda = config.swt.ewc_lambda  # 5000

        # Micro-critics — one per layer
        self.micro_critics = []
        for i, layer in enumerate(model.layers):
            critic = MicroCritic(
                input_dim=layer.out_features,
                hidden_dim=min(256, layer.out_features),
                device=device
            )
            self.micro_critics.append(critic)

        # Competence tracking
        self.task_counter = 0
        self.sleep_frequency = config.swt.frequency_tasks  # 100

    def wake_step(self, x, labels, task_reward):
        """Wake phase: fast TFLE update + store experience.

        Called after each completed task.
        """
        # 1. Run TFLE training step (same as normal training)
        temperature = self.scheduler.get_temperature()
        metrics = self.model.train_step(x, temperature, labels)

        # 2. Compute surprise (for replay priority)
        with torch.no_grad():
            loss = self.model._compute_task_loss(x, labels)
            predicted_loss = self.loss_predictor(x) if hasattr(self, 'loss_predictor') else loss
            surprise = abs(loss - predicted_loss)

        # 3. Store in replay buffer
        self.replay_buffer.add(x, labels, surprise, task_reward)

        # 4. Check if sleep is needed
        self.task_counter += 1
        if self.task_counter >= self.sleep_frequency:
            self.sleep_phase()
            self.task_counter = 0

        return metrics

    def sleep_phase(self):
        """Sleep phase: consolidation via replay + EWC + micro-critics.

        No new data. Only replay buffer.
        """
        # 1. Compute Fisher information for current weights (EWC)
        self._compute_fisher()

        for step in range(self.config.swt.consolidation_steps):  # 500
            # 2. Sample from replay buffer (surprise-priority)
            batch = self.replay_buffer.sample(batch_size=64)
            x, labels, _, _ = batch

            # 3. TFLE consolidation: flips evaluated with EWC penalty
            for layer_idx, layer in enumerate(self.model.layers):
                candidates = layer._select_candidates(layer._get_combined_traces())
                proposals_K = generate_k_proposals(
                    layer.weights, candidates,
                    K=self.config.num_parallel_proposals,
                    device=self.device
                )

                # Evaluate with EWC penalty
                losses = self._eval_with_ewc(layer_idx, proposals_K, x, labels)
                best_k = losses.argmin()
                if losses[best_k] < self.model._compute_task_loss(x, labels):
                    layer.apply_proposal(proposals_K[best_k])

            # 4. Micro-critic consolidation
            for layer_idx, (layer, critic) in enumerate(
                zip(self.model.layers, self.micro_critics)
            ):
                activations = self._get_layer_activations(layer_idx, x)
                critic.train_step(activations)

                # Identify weak weights via critic score
                critic_score = critic.evaluate(activations)
                if critic_score < critic.quality_threshold:
                    # Additional TFLE flips targeting low-scoring activations
                    self._targeted_flip(layer_idx, activations, critic)

        # 5. Adversarial rounds: generate hard examples from past failures
        for _ in range(self.config.swt.adversarial_rounds):  # 3
            hard_batch = self.replay_buffer.sample_hardest(batch_size=32)
            self.model.train_step(hard_batch.x, temperature=0.01, labels=hard_batch.labels)

    def _compute_fisher(self):
        """Compute diagonal Fisher information matrix for EWC."""
        for layer_idx, layer in enumerate(self.model.layers):
            fisher = torch.zeros_like(layer.weights, dtype=torch.float32)
            # Sample from replay buffer
            for _ in range(100):
                batch = self.replay_buffer.sample(batch_size=32)
                x, labels, _, _ = batch
                # Approximate Fisher via loss sensitivity
                base_loss = self.model._compute_task_loss(x, labels)
                for idx in range(min(1000, layer.weights.numel())):
                    # Flip weight, measure loss change
                    old_val = layer.weights.flatten()[idx].item()
                    new_val = (old_val + 2) % 3 - 1
                    layer.weights.flatten()[idx] = new_val
                    new_loss = self.model._compute_task_loss(x, labels)
                    fisher.flatten()[idx] += (new_loss - base_loss) ** 2
                    layer.weights.flatten()[idx] = old_val  # revert
            self.fisher_info[layer_idx] = fisher / 100

    def _eval_with_ewc(self, layer_idx, proposals_K, x, labels):
        """Evaluate proposals with EWC penalty for protecting important weights."""
        base_losses = batched_task_loss_eval(
            self.model, layer_idx, proposals_K, x, labels
        )
        # EWC penalty: penalize changes to high-Fisher weights
        if layer_idx in self.fisher_info:
            fisher = self.fisher_info[layer_idx]
            current_w = self.model.layers[layer_idx].weights.float()
            for k in range(proposals_K.shape[0]):
                diff = (proposals_K[k].float() - current_w) ** 2
                ewc_penalty = (fisher * diff).sum() * self.ewc_lambda
                base_losses[k] += ewc_penalty
        return base_losses


class ReplayBuffer:
    """FOREVER-style replay buffer with surprise-priority sampling."""

    def __init__(self, max_size=10000, priority='surprise'):
        self.max_size = max_size
        self.priority = priority
        self.buffer = []
        self.priorities = []

    def add(self, x, labels, surprise, reward):
        entry = (x.detach().cpu(), labels.detach().cpu(), surprise, reward)
        if len(self.buffer) >= self.max_size:
            # Remove lowest-priority entry
            min_idx = torch.tensor(self.priorities).argmin().item()
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)
        self.buffer.append(entry)
        self.priorities.append(surprise)

    def sample(self, batch_size=64):
        probs = torch.tensor(self.priorities)
        probs = probs / probs.sum()
        indices = torch.multinomial(probs, min(batch_size, len(self.buffer)), replacement=False)
        batch = [self.buffer[i] for i in indices]
        x = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return x, labels, None, None

    def sample_hardest(self, batch_size=32):
        """Sample the highest-surprise experiences."""
        probs = torch.tensor(self.priorities)
        _, indices = probs.topk(min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]
        x = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return x, labels


class MicroCritic(nn.Module):
    """Tiny adversarial network per layer for sleep consolidation.

    Learns to distinguish "good representations" (structured, efficient)
    from "bad representations" (random, redundant).

    Architecture: 100-1000 parameters total.
    """

    def __init__(self, input_dim, hidden_dim=64, device='cpu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.quality_threshold = 0.5
        self.good_buffer = []  # Store high-quality activation snapshots
        self.device = device

    def evaluate(self, activations):
        """Score representation quality (0 = bad, 1 = good)."""
        with torch.no_grad():
            return self.net(activations.detach()).mean().item()

    def train_step(self, activations):
        """Train critic: real activations = positive, noise = negative."""
        batch_size = activations.shape[0]
        noise = torch.randn_like(activations)

        real_scores = self.net(activations.detach())
        fake_scores = self.net(noise)

        loss = -torch.log(real_scores + 1e-8).mean() - torch.log(1 - fake_scores + 1e-8).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

---

## PART 5: REMAINING MISSING COMPONENTS (REFERENCE SPECS)

These come later but the CC session should know what they are. Full implementation details are in `NOVA_BUILD_PLAN.md` Steps 1-25.

### 5A: Full NOVA Model Architecture

Build order (from NOVA_BUILD_PLAN.md):

1. **`model/bitlinear.py`** — Ternary linear layer. ALL internal linear layers use this. Formula: `W_q = clip(round(W / mean(|W|)), -1, 1)`. Activations quantized to 8-bit with absmax. SubLN (RMSNorm) before quantization. ReLU² activation after.

2. **`model/mamba_block.py`** — Selective State Space Model. O(n) sequence length. `Input → Linear(expand) → Conv1D → SiLU → SSM → Linear(project) → Output`. d_state=16, d_conv=4, expand_factor=2. All linear layers inside use BitLinear.

3. **`model/attention.py`** — Grouped-Query Attention with RoPE. 20 query heads, 5 KV heads (4:1 ratio). head_dim=128. RoPE theta=500,000. Q/K/V/O projections use BitLinear.

4. **`model/transformer_block.py`** — `RMSNorm → GQA → Residual → RMSNorm → FFN → Residual`. FFN uses ReLU² (NOT SwiGLU).

5. **`model/hybrid_layer.py`** — Build 32-layer stack: `[M, M, M, A, M, M, M, A, ...]` (1:3 attention-to-Mamba ratio).

6. **`model/nova_model.py`** — Full assembly: Token embedding (NOT quantized) → 32 hybrid layers → RMSNorm → LM head (BitLinear). 2.4B params total. ~0.4GB on disk.

7. **`model/molora.py`** — Per-token LoRA routing. 5 experts: math, code, planning, self_eval, tool_use. rank=8, alpha=16. Top-2 activated per token. Router hidden=256.

### 5B: Training Pipeline

8. **`training/pretrain.py`** — Phase 1: FineWeb-Edu (1.3T tokens), batch=32, lr=3e-4, cosine warmup, BF16→ternary transition at 30%.

9. **`training/distill.py`** — Phase 2: 800K reasoning traces from DeepSeek-R1. SFT on `<think>...</think><answer>...</answer>` format. LoRA rank=16.

10. **`training/rewards.py`** — Binary correctness: math via sympy equivalence check, code via test execution. Format reward: +0.5 for proper think/answer tags.

11. **`training/grpo.py`** — Phase 3: Dr. GRPO. Group size=16 samples per question. Clip ratio=10. KL coeff=0.001. QLoRA 4-bit. 2000 steps.

### 5C: Inference / Intelligence Strategies

12. **`inference/execution_verify.py`** — Strategy 1: Execute code → check → fix → retry (max 5 attempts). Fresh start after 5 failures. Auto-generate tests if none provided. Sandbox: subprocess with 30s timeout.

13. **`inference/consensus.py`** + **`inference/tree_search.py`** — Strategy 2: Tier 1 = self-consistency (8 samples, majority vote). Tier 2 = REBASE tree search (PRM-guided, beam_width=4). Tier 3 = Forest-of-Thought (4 trees with different strategies, 75% consensus threshold).

14. **`inference/prm.py`** — Process Reward Model (~500M params, separate model, can be full-precision). Scores each reasoning step for tree search.

15. **`inference/curiosity.py`** + **`inference/replay_buffer.py`** — Strategy 3: RND novelty (fixed target net + learned predictor). Competence map (15 skill domains, rolling window of 100 attempts each). Learning Progress = current success rate - past success rate. Auto-generate practice tasks at 30-70% expected success.

16. **`inference/tool_orchestrator.py`** — Strategy 4: Probe environment on startup (Python, packages, filesystem, internet, git, docker, browser). Build capability manifest. ReAct loop (max 15 iterations): Thought → Action → Observation. Metacognition: confidence tracking (threshold 0.4), loop detection (3 similar thoughts → switch strategy).

17. **`inference/adversarial_review.py`** — Strategy 5: Structured checklists ONLY (NOT open-ended critique — research shows 1-3B models DEGRADE with open-ended self-review). Separate checklists for code, general, and reasoning. Each item is a specific yes/no question. Max 3 review rounds.

18. **`inference/difficulty_router.py`** — Route by model confidence: Easy (1 sample, no review) → Medium (4 samples, 1 review) → Hard (8 samples + tree search, 2 reviews) → Extreme (16 samples + FoT, 3 reviews).

### 5D: Agent Loop

19. **`agent/nova_agent.py`** — Main loop: parse task → check tools → estimate difficulty → plan → generate → verify → review → learn → return. Wake update after every task. Sleep every 100 tasks.

20. **`agent/task_parser.py`** — Parse prompts into structured tasks (type, requirements, constraints).

21. **`agent/planner.py`** — Decompose complex tasks into verifiable sub-steps.

22. **`agent/environment.py`** — Filesystem, terminal, browser, API state tracking.

23. **`agent/memory.py`** — Persistent key-value store across sessions.

### 5E: Benchmarks

24. **`benchmarks/run_benchmarks.py`** — Primary: MATH-500, HumanEval, MBPP, GSM8K. Secondary: MMLU, ARC-C, BBH, GPQA-Diamond, AIME-2024. Agent: SWE-Bench-Verified, LiveCodeBench. Ablation: each strategy disabled one at a time.

---

## BUILD PRIORITY ORDER

```
IMMEDIATE (this session / next session):
  1. GPU device support (Part 1, Step 1.1)
  2. Batched parallel evolution (Part 1, Steps 1.2-1.4)
  3. Test: MNIST 85%+ with GPU acceleration

NEXT:
  4. CDLL compression fitness (Part 2)
  5. Mono-Forward local classifier heads (Part 3)
  6. Test: Compare task-loss vs CDLL vs Mono-Forward vs hybrid on MNIST
  7. Test: Layer-parallel training with local fitness on GPU

THEN:
  8. SWT sleep-wake cycle (Part 4)
  9. Test: Train MNIST → CIFAR-10 sequentially, measure forgetting

AFTER TFLE IS PROVEN:
  10. Full model architecture (Part 5A) — BitLinear, Mamba, Attention, Hybrid, MoLoRA
  11. Pretraining pipeline (Part 5B)
  12. Inference strategies (Part 5C) — start with execution_verify (highest impact)
  13. Agent loop (Part 5D)
  14. Benchmarks (Part 5E)
```

---

## KEY WARNINGS FOR CC

1. **BitLinear is the foundation.** Every linear layer (except embedding/LM head) uses BitLinear. If you use `nn.Linear` inside the model, it's wrong.

2. **ReLU², not SwiGLU.** Standard transformers use SwiGLU. NOVA uses ReLU² (squared ReLU). Provides stronger sparsity for ternary.

3. **Transformer bug.** `transformer.py` line ~223 uses `torch.randn()` instead of real activations. Fix before using.

4. **fitness.py is disconnected.** Three fitness functions exist but aren't wired into anything. Wire them up as part of CDLL integration.

5. **Small models can't self-correct from prompts.** Don't build open-ended self-critique. Use structured checklists + execution verification. Self-correction comes from RL (GRPO), not inference-time prompting.

6. **TFLE has no precedent.** All ternary training uses STE. TFLE is genuinely novel. Build it but also keep the STE baseline for comparison.

7. **The three pillars must compose.** CDLL provides the fitness function, TFLE provides the update mechanism, SWT provides the training schedule. They're designed to work as a stack:
   - Wake phase: TFLE flips guided by CDLL fitness
   - Sleep phase: TFLE flips on replay buffer with EWC protection, evaluated by micro-critics + CDLL

8. **Test each module independently.** Every file should have tests. Don't move to the next module until the current one works.

9. **Device management is non-negotiable.** Every tensor must be on the correct device. Every new tensor created inside a function must inherit the device from its inputs. Use `tensor.device` to stay consistent.

---

*Created: April 1, 2026*
*Status: Ready for implementation — start with GPU acceleration (Part 1)*
*Context docs: NOVA_CC_BRIEFING.md, NOVA_SYSTEM_SPEC.md, NOVA_BUILD_PLAN.md*
*Theory docs: compression-driven-learning.md, sleep-wake-training.md, trit-flip-local-evolution.md*
