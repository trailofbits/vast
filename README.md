[![Build](https://github.com/trailofbits/vast/actions/workflows/build-test-release.yml/badge.svg)](https://github.com/trailofbits/vast/actions/workflows/build-test-release.yml)

# VAST — Verbose AST

VAST is an experimental frontend for the translation of Clang AST to various MLIR dialects. These dialects allow exploring the codebase at a specific level of abstraction before reaching the low-level LLVM dialect.

## Build

To configure project run:

```
cmake --preset ninja-multi-default \
      -DLLVM_INSTALL_DIR=<llvm instalation path>
```

Finally build and install the binaries:

```
cmake --build --preset ninja-release
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
