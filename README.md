[![Build](https://github.com/trailofbits/vast/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/trailofbits/vast/actions/workflows/main.yml)

# VAST — Verbose AST

VAST is an experimental frontend for the translation of Clang AST to various MLIR dialects. These dialects allow exploring the codebase at a specific level of abstraction before reaching the low-level LLVM dialect.

## Build

To configure project run:

```
cmake \
    -S . -B build \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX=<install directory> \
    -DLLVM_INSTALL_DIR=<llvm instalation path>
```

If you want to build with tests:

```
cmake \
    -S . -B build \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX=<install directory> \
    -DLLVM_INSTALL_DIR=<llvm instalation path> \
    -DENABLE_TESTING=ON \
    -DLLVM_EXTERNAL_LIT=<path to lit>
```

Finally build and install the binaries:

```
cmake --build build --target install
```

## Run

To run mlir codegen of highlevel dialect use:

```
./build/bin/vast-cc --from-source <input.c>
```

## Test

```
cmake --build <build-dir> --target check-vast
```
