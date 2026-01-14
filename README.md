# mruntime

`mruntime` is a minimal, bare-metal LLM inference runtime built for systems researchers who want a small, hackable codebase they can modify quickly.

Inspired by [OpenBMB/CPM.cu](https://github.com/OpenBMB/CPM.cu), mruntime prioritizes simplicity and directness over abstraction. No virtual dispatch, no runtime dtype switching, no deep class hierarchies—just raw pointers, free functions, and explicit data flow.

Current focus: Qwen2-family decoder-only models (e.g., Qwen2.5) on Arm64 CPU.

## Features

- **Bare-metal API**: Free functions with raw pointers, no `Tensor` wrapper in hot paths
- **Arena allocation**: Pre-allocated memory pools for weights, KV cache, and scratch buffers
- **FP16 hardcoded**: No runtime dtype dispatch (BF16 weights converted at load time)
- **Flat call stack**: 2-3 levels from entry point to kernel
- **Arm64 GEMM fast-path**: KleidiAI integration for optimized matrix multiplication
- **Simple chat CLI**: `mruntime_chat` for local experimentation

## Non-goals

- Many model architectures (MoE, Vision, BERT, etc.)
- Many quantization formats
- Many backends (GPU/NN accelerators)
- Framework-style abstractions

## Build

```bash
cmake -S . -B build -DMRUNTIME_BUILD_TESTS=ON -DMRUNTIME_BUILD_BENCH=ON -DMRUNTIME_BUILD_CLI=ON
cmake --build build -j
```

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Model-dependent tests require a Qwen2.5 model directory at `models/Qwen2.5-0.5B-Instruct/`.

## CLI

```bash
./build/mruntime_chat --model-dir models/Qwen2.5-0.5B-Instruct
```

Flags:
- `--greedy` for deterministic decoding
- `--temperature`, `--top-k`, `--top-p` for sampling
- `--max-new-tokens N` to limit generation length

## Architecture

Flat, function-based design:

```
include/mruntime/
├── arena.h              # Memory arena (bump allocator)
├── qwen2_ops.h          # Core FP16 operations
├── qwen2_weights.h      # Weight structures + memory sizing
├── qwen2_forward.h      # Forward pass functions
├── qwen2_generate.h     # Generation loop + sampling

src/
├── qwen2_ops.cpp        # RMSNorm, RoPE, attention, GEMM, etc.
├── qwen2_weights.cpp    # Weight loading into arena
├── qwen2_forward.cpp    # Model forward pass
├── qwen2_generate.cpp   # Generation logic
├── core/kai_gemm.cpp    # KleidiAI GEMM integration
```

## API Example

```cpp
// Allocate arenas
Qwen2MemorySizes sizes = qwen2_memory_sizes(cfg, max_seq_len);
Qwen2Arenas arenas = create_qwen2_arenas(sizes.weights_bytes,
                                          sizes.kv_cache_bytes,
                                          sizes.scratch_bytes);

// Load weights into arena
auto file = SafeTensorsFile::open(model_path);
Qwen2Weights weights = qwen2_load_weights(cfg, *file, arenas.weights);

// Initialize state
Qwen2KVCache kv = qwen2_init_kv_cache(cfg, arenas.kv_cache, max_seq_len);
Qwen2Scratch scratch = qwen2_init_scratch(cfg, arenas.scratch);

// Generate
PThreadPool pool = PThreadPool::Create(0);
size_t len = qwen2_generate(cfg, weights, kv, scratch,
                            prompt, prompt_len, output, gen_cfg, &pool);

// Cleanup
destroy_qwen2_arenas(arenas);
```

## Design Principles

1. **Raw pointers + explicit sizes** — No wrapper classes in hot paths
2. **Free functions** — `qwen2_forward()` not `model.forward()`
3. **Arena allocation** — Pre-allocated pools, bump allocation
4. **FP16 hardcoded** — No dtype dispatch per element
5. **Flat call stack** — Easy to trace, profile, and modify

## Documentation

See [docs/REFACTORING.md](docs/REFACTORING.md) for detailed refactoring notes.

## License

Research prototype. See LICENSE file for details.
