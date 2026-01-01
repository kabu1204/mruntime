# MLAS build + benchmark notes

This repo vendors MLAS under `MLAS/` and includes a simple benchmark driver at the repo root: `main.cpp`.

## Benchmark (`main.cpp`)

The benchmark times:
- FP32 GEMM via `MlasGemm`
- FP32 FlashAttention via `MlasFlashAttention`

It uses `std::chrono::steady_clock` and prints average milliseconds across a small number of iterations.

### Build & run (example)

If you built and installed MLAS into `MLAS/install`:

```sh
c++ -O3 -std=c++17 -I MLAS/include main.cpp \
  -L MLAS/install/lib -lonnxruntime_mlas -lonnxruntime_common \
  -o mlas_bench

./mlas_bench
```

Notes:
- If `onnxruntime_common` is not installed yet, you can link it from the build tree: `-L MLAS/build/src -lonnxruntime_common`.
- If you configured MLAS with `-DMLAS_NO_ONNXRUNTIME=ON`, you can usually drop `-lonnxruntime_common` (standalone build).

## Building MLAS with system Eigen (Homebrew)

MLAS normally fetches Eigen. To use a system Eigen install instead:

```sh
mkdir -p MLAS/build && cd MLAS/build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DEigen3_DIR="$(brew --prefix eigen)/share/eigen3/cmake"
make -j
make install
```

If you only want to provide the include directory:

```sh
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DEIGEN3_INCLUDE_DIR="$(brew --prefix eigen)/include/eigen3"
```

The detection logic lives in `MLAS/cmake/external_deps.cmake`. It only FetchContent-downloads Eigen if a system Eigen cannot be found.

## macOS build error: `namespace timestamp_ns = ::date;`

If you see:

```txt
.../logging.h:91:28: error: expected namespace name
namespace timestamp_ns = ::date;
```

That fallback expects the Howard Hinnant `date` library (`namespace date`) for timestamp formatting (used by ONNX Runtime on some platforms/configs).
This standalone MLAS build did not provide the dependency, so a minimal shim header was added at `MLAS/src/ort_include/date/date.h` and included from
`MLAS/src/ort_include/core/common/logging/logging.h`.

Alternative (avoid the `date` path): configure with a deployment target new enough to keep the `std::chrono` formatting path, e.g.
`-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3`.

## `make install` failing due to google benchmark

If `make install` fails with:

```txt
include could not find requested file: .../_deps/google_benchmark-build/cmake_install.cmake
```

The fix is to include google benchmark via plain `FetchContent_MakeAvailable(...)` so the install script is generated.
This is handled in `MLAS/cmake/external_deps.cmake`. After changing CMake logic, re-run `cmake ..` (and wipe `MLAS/build` if the old state is cached).

## Linking note: `onnxruntime_common`

By default (`MLAS_NO_ONNXRUNTIME=OFF`), MLAS compiles a small subset of ONNX Runtime support code (as `onnxruntime_common`) and `libonnxruntime_mlas.a`
references it (e.g. `onnxruntime::CPUIDInfo`, `onnxruntime::concurrency::ThreadPool`).

Fixes if you get undefined symbols while linking `main.cpp`:
- Link both `-lonnxruntime_mlas -lonnxruntime_common`, or
- Configure MLAS with `-DMLAS_NO_ONNXRUNTIME=ON` and rebuild/install.

## What is `onnxruntime_mlas_q4dq`?

`onnxruntime_mlas_q4dq` is a small CLI tool built from `MLAS/src/lib/q4_dq_cli.cpp` that quantizes/dequantizes 2-D FP32 tensors using MLAS’s blockwise int4 (Q4) format.

