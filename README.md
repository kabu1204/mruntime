# mruntime

`mruntime` is a minimal, research-friendly on-device LLM runtime built for systems researchers who want a small codebase they can modify quickly.

Many existing inference stacks are heavy and tightly coupled (many model families, quantization formats, and backends), which makes research iteration painful. `mruntime` intentionally keeps scope narrow and code easy to read.

Current focus: Qwen2-family decoder-only models (e.g., Qwen2.5) on Arm64 CPU.

## Features

- CPU-only inference with a small, explicit `Backend` abstraction.
- Primary runtime format is FP16 (activations / KV-cache / logits default to FP16).
- Weights may be FP16 or BF16 (as stored in the model); conversions happen as needed.
- Arm64 GEMM fast-path via KleidiAI.
- Simple chat CLI (`mruntime_chat`) for local experimentation.

## Non-goals (Today)

- Many model architectures (MoE, Vision, BERT, etc.).
- Many quantization formats.
- Many backends (GPU/NN accelerators).

## Build

Out-of-source build (recommended):

```bash
cmake -S . -B build -DMRUNTIME_BUILD_TESTS=ON -DMRUNTIME_BUILD_BENCH=ON -DMRUNTIME_BUILD_CLI=ON
cmake --build build -j
```

## Tests

Run the full test suite:

```bash
ctest --test-dir build --output-on-failure
```

Model-free tests can be run directly:

```bash
./build/test/tensor_test
./build/test/backend_test
./build/test/tokenizer_test
```

Model-dependent tests require a Qwen2.5 model directory at `models/Qwen2.5-0.5B-Instruct/` (weights are typically gitignored, but config/tokenizer files should be present):

- `./build/test/loader_test`
- `./build/test/model_test`
- `./build/test/engine_test`
- `./build/test/qwen2_tokenizer_test`

Note: some model-dependent tests assume working-directory == the build directory.

## CLI

Build with `-DMRUNTIME_BUILD_CLI=ON`, then run:

```bash
./build/mruntime_chat --model-dir models/Qwen2.5-0.5B-Instruct
```

Useful flags:
- `--greedy` for deterministic decoding
- `--temperature`, `--top-k`, `--top-p` for sampling
- `--profile-backend` for per-op timing summary
- `--profile-backend-trace PATH` to write a Chrome trace JSON

## Architecture (High Level)

Layered design from bottom to top:

1. **Tensor** (`include/mruntime/tensor.h`, `src/core/tensor.cpp`)
2. **Backend** (`include/mruntime/backend.h`, `src/core/cpu_backend.cpp`)
3. **Model** (`include/mruntime/qwen_model.h`, `src/models/qwen/qwen_model.cpp`)
4. **Engine** (`include/mruntime/inference_engine.h`, `src/engine/inference_engine.cpp`)

## Known Blockers and Risks

This is a research prototype with known limitations:

- **KV-cache assumptions**: code may assume batch=1 and/or specific dtypes/layouts without strong validation.
- **Repeated dtype conversions**: some hot paths convert large tensors repeatedly (e.g., via `ensure_fp32()`), which is slow.
- **Full-tensor transposes**: some layout changes use full copies (e.g., `permute()` is not a view).

These are documented risks; treat them as research tradeoffs rather than production bugs.

## Future Directions

The project is structured to make future extensions possible without committing to them today:

- additional backends (GPU / heterogeneous CPU+GPU)
- common quantization formats
- alternative kernels behind the `Backend` abstraction
