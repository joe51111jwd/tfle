# NOVA Multi-GPU Scaling — CC Implementation Outline

**Target**: Dual RTX 5090 (32GB VRAM each, 64GB total)
**Constraint**: Preserve ALL existing training logic — same fitness functions, same accept/reject, same trace updates, same annealing. Only the *execution topology* changes.

---

## Background: What We Have Now

The codebase (commit `e03c636`) runs on a single GPU. Key facts:

- `resolve_device()` returns one `torch.device("cuda")` — always GPU 0
- `TFLEModel.__init__` puts all layers on `self.device` (one device)
- `_train_step_local()` loops over layers sequentially, but each layer's fitness is independent (CDLL/Mono-Forward only depend on that layer's input/output)
- `_train_step_task_loss_batched()` needs full-model forward passes — inherently sequential across layers
- `generate_k_proposals()` is already vectorized on GPU
- `ternary_matmul()` casts int8 to float and does `x @ weights.float()` — no INT8 tensor core usage

## Part 1: Multi-Device Config & Device Map

**File**: `tfle/config.py`

Add multi-GPU configuration parameters to `TFLEConfig`:

```python
# --- 17. Multi-GPU Scaling ---
multi_gpu: bool = False                    # Enable multi-GPU mode
gpu_devices: list[int] = field(default_factory=lambda: [0, 1])  # GPU IDs to use
layer_device_map: str = "auto"             # "auto" | "round_robin" | "memory_balanced" | "manual"
manual_device_map: dict[int, int] = field(default_factory=dict)  # layer_idx -> gpu_id (for manual)
cross_gpu_sync_interval: int = 1           # How often to sync layer inputs across GPUs
data_parallel_task_loss: bool = True       # Use data parallelism for task-loss mode
prefetch_layer_inputs: bool = True         # Pre-compute and cache layer inputs on each device
```

**New function**: `build_device_map(config, layer_sizes) -> dict[int, torch.device]`

```
auto strategy:
  - Count params per layer (in_f * out_f)
  - Greedily assign layers to GPUs to balance total params per GPU
  - Earlier layers tend to be bigger (e.g., 784x512 vs 256x10), so auto balances this

round_robin strategy:
  - Layer 0 -> GPU 0, Layer 1 -> GPU 1, Layer 2 -> GPU 0, ...

memory_balanced strategy:
  - Estimate memory per layer: weights (int8) + traces (float16 x2) + proposals (K * in * out * int8)
  - K proposals are the memory-dominant term during training
  - Assign layers to minimize max memory per GPU
```

Returns `{0: torch.device("cuda:0"), 1: torch.device("cuda:1"), 2: torch.device("cuda:0"), ...}`

**Update `resolve_device()`**: When `multi_gpu=True`, return `torch.device("cuda:0")` as the "primary" device. The device map handles per-layer placement.

---

## Part 2: Per-Layer Device Placement

**File**: `tfle/layers.py`

### 2a. TFLELayer gets its own device

`TFLELayer.__init__` already takes a `device` param. No change needed to the constructor — the caller (TFLEModel) will pass `device_map[layer_idx]` instead of `self.device`.

Key: weights, traces, cooldown state all live on the layer's assigned GPU. The `forward()` call already uses `weights.to(dtype=x.dtype, device=x.device)` which handles cross-device input, but we want to avoid that cast overhead by ensuring inputs arrive on the correct device.

### 2b. generate_k_proposals respects layer device

`generate_k_proposals()` takes a `device` param. When called, pass `layer.device` instead of `self.device`. Proposals are generated and evaluated entirely on the layer's GPU — no cross-GPU transfer needed.

### 2c. ternary_matmul stays the same

No changes. It already handles device placement via `.to(device=x.device)`. When layer and input are on the same GPU, this is a no-op.

---

## Part 3: TFLEModel Multi-Device Init & Forward

**File**: `tfle/model.py`

### 3a. __init__ with device map

```python
def __init__(self, config, device=None):
    ...
    if config.multi_gpu and torch.cuda.device_count() >= 2:
        self.device_map = build_device_map(config, config.layer_sizes)
        self.multi_gpu = True
        self.device = torch.device("cuda:0")  # primary device for final output
    else:
        self.device_map = None
        self.multi_gpu = False
        self.device = resolve_device(config)

    for i, (in_f, out_f) in enumerate(zip(config.layer_sizes[:-1], config.layer_sizes[1:])):
        layer_device = self.device_map[i] if self.multi_gpu else self.device
        self.layers.append(TFLELayer(in_f, out_f, config, layer_idx=i, device=layer_device))

    # CDLL fitness evaluators — same device as their layer
    if config.fitness_type in (FitnessType.CDLL, FitnessType.HYBRID_LOCAL):
        for i, (in_f, out_f) in enumerate(zip(...)):
            dev = self.device_map[i] if self.multi_gpu else self.device
            self.cdll_fitness.append(CDLLFitness(in_f, out_f, i, config, dev))

    # Local heads — same device as their layer
    if config.fitness_type in (FitnessType.MONO_FORWARD, FitnessType.HYBRID_LOCAL):
        for i, (_, out_f) in enumerate(zip(...)):
            dev = self.device_map[i] if self.multi_gpu else self.device
            self.local_heads.append(LocalClassifierHead(out_f, num_classes, config, dev))
```

### 3b. forward() with cross-device transfers

```python
def forward(self, x):
    x = x.to(self.device)
    for i, layer in enumerate(self.layers):
        if self.multi_gpu:
            x = x.to(layer.device)  # transfer to layer's GPU if needed
        x = layer.forward(x)
        if i < len(self.layers) - 1:
            x = F.relu(x)
    if self.multi_gpu:
        x = x.to(self.device)  # final output on primary device
    return x
```

Cross-device transfer cost: for a 2.4B model with layers like `[784, 2048, 2048, 2048, 2048, 1024, 512, 10]`, the intermediate activations are `batch_size * hidden_dim * 4 bytes` (float32). At batch=64, hidden=2048, that's 512KB per transfer — negligible compared to the compute cost of the matmul + proposal evaluation.

### 3c. _compute_layer_inputs() with device-aware caching

```python
def _compute_layer_inputs(self, x):
    inputs = [x.to(self.layers[0].device) if self.multi_gpu else x]
    h = inputs[0]
    with torch.no_grad():
        for i, layer in enumerate(self.layers):
            if self.multi_gpu:
                h = h.to(layer.device)
            h = layer.forward(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
            # Store on the NEXT layer's device (that's who needs it)
            if self.multi_gpu and i + 1 < len(self.layers):
                inputs.append(h.to(self.layers[i + 1].device))
            else:
                inputs.append(h)
    return inputs
```

This ensures each layer's input is already on the correct GPU when training begins.

---

## Part 4: Layer-Parallel Local Training (THE BIG WIN)

**File**: `tfle/model.py` — `_train_step_local()`

This is where dual 5090s pay off. Currently this is a sequential `for` loop over layers. With local fitness (CDLL/Mono-Forward/Hybrid), each layer's fitness depends ONLY on its own input and output — layers are independent. So layers on different GPUs can train simultaneously.

### Strategy: CUDA Streams

Each GPU gets its own CUDA stream. Layers assigned to the same GPU execute sequentially within that stream. Layers on different GPUs execute concurrently.

```python
def _train_step_local(self, x, labels, temperature, mode):
    K = max(1, self.config.num_parallel_proposals)
    all_metrics = [None] * len(self.layers)

    # Phase 1: Cache all layer inputs (must be sequential — each depends on previous)
    layer_inputs = self._compute_layer_inputs(x)
    # Also cache labels on each device
    if self.multi_gpu:
        labels_per_device = {}
        for dev in set(self.device_map.values()):
            labels_per_device[dev] = labels.to(dev)

    # Phase 2: Train layers in parallel across GPUs using CUDA streams
    if self.multi_gpu:
        streams = {}
        for dev in set(self.device_map.values()):
            streams[dev] = torch.cuda.Stream(device=dev)

        # Launch all layers
        for layer_idx, layer in enumerate(self.layers):
            dev = self.device_map[layer_idx]
            stream = streams[dev]
            with torch.cuda.stream(stream):
                layer_labels = labels_per_device[dev]
                metrics = self._train_single_layer_local(
                    layer_idx, layer, layer_inputs[layer_idx],
                    layer_labels, temperature, mode, K
                )
                all_metrics[layer_idx] = metrics

        # Sync all streams
        for stream in streams.values():
            stream.synchronize()
    else:
        # Single GPU: sequential (current behavior)
        for layer_idx, layer in enumerate(self.layers):
            metrics = self._train_single_layer_local(
                layer_idx, layer, layer_inputs[layer_idx],
                labels, temperature, mode, K
            )
            all_metrics[layer_idx] = metrics

    return all_metrics
```

### Extract `_train_single_layer_local()`

Pull the body of the current per-layer loop into its own method. This is a pure refactor — the logic inside is identical:

```python
def _train_single_layer_local(self, layer_idx, layer, layer_in, labels, temperature, mode, K):
    """Train one layer with local fitness. Self-contained, no cross-layer deps."""
    layer.step_count += 1

    # Decay cooldowns (unchanged)
    ...

    # Current output + fitness (unchanged)
    with torch.no_grad():
        current_out = layer.forward(layer_in)
        if layer_idx < len(self.layers) - 1:
            current_out_act = F.relu(current_out)
        else:
            current_out_act = current_out
    fitness_before = self._local_fitness(layer_idx, layer_in, current_out_act, labels, mode)

    # Select candidates, generate K proposals (unchanged — runs on layer.device)
    combined_traces = layer._get_combined_traces()
    candidates = layer._select_candidates(combined_traces)
    if K > 1:
        proposals = generate_k_proposals(layer.weights, candidates, K, layer.device)
    else:
        proposals = layer._propose_flips(candidates).unsqueeze(0)

    # Evaluate proposals with local fitness (unchanged)
    best_fitness = fitness_before
    best_k = -1
    for k in range(proposals.shape[0]):
        with torch.no_grad():
            w_float = proposals[k].float().to(layer.device)
            out_k = layer_in @ w_float
            if layer_idx < len(self.layers) - 1:
                out_k = F.relu(out_k)
        f_k = self._local_fitness(layer_idx, layer_in, out_k, labels, mode)
        if f_k > best_fitness:
            best_fitness = f_k
            best_k = k

    # Accept/reject with Boltzmann (unchanged)
    delta = best_fitness - fitness_before
    layer_temp = self.config.get_temperature_for_layer(temperature, layer_idx)
    accepted = layer._accept_or_reject(delta, layer_temp)

    if accepted and best_k >= 0:
        layer.weights = proposals[best_k].to(torch.int8)

    # Update local head, update traces (unchanged)
    if mode in ("mono_forward", "hybrid_local") and self.local_heads:
        self.local_heads[layer_idx].update(current_out_act, labels)
    output = layer.forward(layer_in)
    error_signal = delta <= 0
    layer._update_traces(layer_in, output, error_signal)

    layer.fitness_history.append(best_fitness if accepted else fitness_before)
    layer.acceptance_history.append(accepted)

    return { ... }  # same metrics dict as before
```

### Why This Is Safe

Local fitness only reads `layer_in` (cached before training) and the layer's own weights/proposals. No layer reads or writes another layer's state. The accept/reject logic, trace updates, cooldown maps — all layer-local. Two layers on two GPUs are fully independent.

### Expected Speedup

With dual 5090s and a ~8-layer NOVA model split 4+4:
- Current: 8 sequential layer steps per training step
- Multi-GPU: 4 sequential steps per GPU, both GPUs in parallel = ~2x throughput
- Real-world: ~1.6-1.8x after accounting for the sequential `_compute_layer_inputs()` pass and stream sync overhead

---

## Part 5: Data-Parallel Task-Loss Mode

**File**: `tfle/model.py` — `_train_step_task_loss_batched()`

Task-loss requires full-model forward passes, so layer-parallelism doesn't apply. Instead, use **data parallelism**: split the batch across GPUs, evaluate proposals on both GPUs simultaneously, reduce losses.

### Strategy

```python
def _train_step_task_loss_batched_multi_gpu(self, x, labels, temperature):
    K = self.config.num_parallel_proposals

    # Split batch across GPUs
    B = x.shape[0]
    half = B // 2
    x_splits = [x[:half].to(self.devices[0]), x[half:].to(self.devices[1])]
    l_splits = [labels[:half].to(self.devices[0]), labels[half:].to(self.devices[1])]

    # For each layer (sequential — task-loss needs full model):
    for layer_idx, layer in enumerate(self.layers):
        ...
        # Generate proposals on layer's device
        proposals = generate_k_proposals(layer.weights, candidates, K, layer.device)
        all_proposals = torch.cat([current_w, proposals], dim=0)

        # Evaluate on both GPUs in parallel via streams
        # GPU 0: batched_task_loss_eval(model, layer_idx, all_proposals, x_splits[0], l_splits[0])
        # GPU 1: batched_task_loss_eval(model, layer_idx, all_proposals, x_splits[1], l_splits[1])
        # Average the losses from both GPUs

        losses = (losses_gpu0 + losses_gpu1) / 2  # mean across data splits
        ...  # rest of accept/reject unchanged
```

This doubles the effective batch size for proposal evaluation without changing the training semantics. The accept/reject decision uses the averaged loss — mathematically equivalent to evaluating on the full batch.

**Note**: This requires a model replica on GPU 1 (just the weights for the forward pass, not the full training state). Alternatively, use `batched_task_loss_eval` which only needs layer weights, not the full model — we can pass weight tensors to GPU 1 on demand.

### When to use which mode

| Fitness Type | Multi-GPU Strategy | Speedup |
|---|---|---|
| CDLL / Mono-Forward / Hybrid | Layer-parallel (Part 4) | ~1.7x |
| Task-Loss (batched) | Data-parallel (Part 5) | ~1.5x |
| Contrastive (legacy) | Data-parallel | ~1.5x |

---

## Part 6: SWT (Sleep-Wake Training) Multi-GPU

**File**: `tfle/swt.py`

### 6a. ReplayBuffer stays on CPU

The replay buffer stores detached CPU tensors (already does `x.detach().cpu()`). This is correct for multi-GPU — CPU memory is shared, and samples are sent to the correct GPU on demand.

### 6b. Sleep phase uses layer-parallel

`sleep_phase()` currently does manual per-layer proposal/evaluation. Refactor to use the same layer-parallel pattern from Part 4:

```python
def sleep_phase(self, temperature):
    ...
    for _ in range(self.config.swt_consolidation_steps):
        batch = self.replay_buffer.sample(batch_size=64)
        x, labels = batch

        # Use layer-parallel if multi-GPU and local fitness
        if self.model.multi_gpu:
            # Send data to each device once
            # Run per-layer consolidation in parallel streams
            ...
        else:
            # Current sequential logic (unchanged)
            ...
```

### 6c. Fisher computation parallelized

`_compute_fisher()` is embarrassingly parallel across layers. Each layer independently measures flip sensitivity. Wrap in the same CUDA streams pattern.

### 6d. Micro-critics stay on their layer's device

Each `MicroCritic` is initialized with a device. When `SleepWakeTrainer.__init__` creates them, use `device_map[layer_idx]` if multi-GPU.

---

## Part 7: Optimized Ternary Matmul (INT8 Tensor Cores)

**File**: `tfle/layers.py` — `ternary_matmul()`

RTX 5090 has INT8 tensor cores (Blackwell architecture). Current code casts to float32 for matmul. For ternary weights {-1, 0, +1} stored as int8, we can use `torch._int_mm` (available since PyTorch 2.1):

```python
def ternary_matmul(x, weights):
    if weights.dtype == torch.int8 and x.is_cuda:
        # INT8 tensor cores: x must also be int8
        # For float input, quantize to int8, multiply, then scale back
        # OR: use torch.float16 matmul which is also much faster than float32
        if x.dtype == torch.float32:
            x_half = x.half()
            w_half = weights.half()
            return (x_half @ w_half).float()
        elif x.dtype == torch.float16:
            return x @ weights.half()
    # Fallback
    return x @ weights.to(dtype=x.dtype, device=x.device)
```

**Better approach for pure ternary**: Since weights are only {-1, 0, +1}, matmul is really just add/subtract/skip. A custom CUDA kernel could do this without any multiplication. But float16 matmul on tensor cores is already very fast and simpler to implement. Start with float16, benchmark, then consider custom kernel if needed.

**Impact**: float16 matmul on RTX 5090 is ~2x faster than float32 for large matrices. For a 2048x2048 layer, this is significant.

---

## Part 8: TFLEModel.to() for Multi-GPU

**File**: `tfle/model.py`

Update `to()` to support multi-GPU:

```python
def to(self, device):
    if isinstance(device, dict):
        # device_map: {layer_idx: torch.device}
        self.device_map = device
        self.multi_gpu = True
        for i, layer in enumerate(self.layers):
            dev = device.get(i, torch.device("cuda:0"))
            layer.weights = layer.weights.to(dev)
            layer.device = dev
            # Move traces
            if self.config.separate_pos_neg_traces:
                layer.success_traces = layer.success_traces.to(dev)
                layer.error_traces = layer.error_traces.to(dev)
            else:
                layer.traces = layer.traces.to(dev)
            # Move CDLL fitness decoders if they exist
            if self.cdll_fitness:
                if self.cdll_fitness[i].decoder is not None:
                    self.cdll_fitness[i].decoder = self.cdll_fitness[i].decoder.to(dev)
            # Move local heads
            if self.local_heads:
                self.local_heads[i].classifier = self.local_heads[i].classifier.to(dev)
        return self
    else:
        # Single device (current behavior)
        ...
```

---

## Part 9: Training Loop Updates

**File**: `tfle/training.py`

### 9a. TFLETrainer handles multi-GPU

```python
class TFLETrainer:
    def __init__(self, model, config, train_loader, val_loader=None, ...):
        ...
        # Validate multi-GPU setup
        if config.multi_gpu:
            n_gpus = torch.cuda.device_count()
            assert n_gpus >= 2, f"multi_gpu=True but only {n_gpus} GPU(s) found"
            for gid in config.gpu_devices:
                assert gid < n_gpus, f"GPU {gid} not available (only {n_gpus} GPUs)"
```

### 9b. Batch placement

When `multi_gpu=True`, batches go to the primary device (GPU 0). The model's `train_step` handles per-layer device transfers internally.

### 9c. NCCL peer-to-peer

Enable P2P transfers between GPUs for faster cross-device tensor moves:

```python
if config.multi_gpu:
    for i in config.gpu_devices:
        for j in config.gpu_devices:
            if i != j:
                torch.cuda.set_device(i)
                torch.cuda.enable_peer_to_peer(i, j)  # if supported
```

On dual 5090s with NVLink or PCIe 5.0, P2P transfers are fast (~50GB/s NVLink, ~32GB/s PCIe 5.0 x16).

---

## Part 10: Checkpoint Save/Load for Multi-GPU

**File**: `tfle/model.py`

```python
def save_checkpoint(self, path):
    state = {
        "config": self.config,
        "weights": [layer.weights.cpu().clone() for layer in self.layers],
        "device_map": self.device_map if self.multi_gpu else None,
    }
    torch.save(state, path)

def load_checkpoint(self, path):
    state = torch.load(path, weights_only=False, map_location="cpu")
    for i, (layer, weights) in enumerate(zip(self.layers, state["weights"])):
        dev = self.device_map[i] if self.multi_gpu else self.device
        layer.weights = weights.to(dev)
```

Always save to CPU, load to the correct device. This makes checkpoints portable between single-GPU and multi-GPU setups.

---

## Implementation Order

1. **Part 1**: Config additions (small, no behavior change)
2. **Part 2**: Per-layer device placement in `TFLELayer` (minimal changes)
3. **Part 3**: `TFLEModel.__init__` and `forward()` with device map
4. **Part 8**: `TFLEModel.to()` multi-GPU support
5. **Part 10**: Checkpoint save/load
6. **Part 4**: Layer-parallel `_train_step_local()` with CUDA streams — **highest impact**
7. **Part 5**: Data-parallel task-loss mode
8. **Part 6**: SWT multi-GPU
9. **Part 7**: Optimized ternary matmul (float16)
10. **Part 9**: Training loop validation + P2P setup

## Testing Strategy

- All existing tests must pass unchanged (single-GPU/CPU mode)
- New tests:
  - `test_device_map_balanced`: verify auto device map balances params
  - `test_multi_gpu_forward`: output matches single-GPU forward (tolerance 1e-6)
  - `test_layer_parallel_equivalence`: local training on 2 GPUs produces same accept/reject decisions as sequential (set same seed)
  - `test_checkpoint_portability`: save multi-GPU, load single-GPU and vice versa
  - `test_data_parallel_task_loss`: loss values match single-GPU (tolerance 1e-5)
  - Mock tests for systems with only 1 GPU (use `cuda:0` as both devices)

## What Does NOT Change

- Candidate selection logic (`_select_candidates`)
- Flip proposal logic (`_propose_flips`, `generate_k_proposals`)
- Accept/reject logic (`_accept_or_reject` — Boltzmann/Metropolis)
- Trace update logic (`_update_traces`)
- Fitness functions (CDLL, Mono-Forward, contrastive, task-loss)
- Temperature annealing schedule
- EWC penalty computation
- Replay buffer sampling
- Micro-critic architecture
- Weight initialization

The training algorithm is identical. Only the execution topology changes: which GPU runs which computation, and whether independent computations overlap in time.
