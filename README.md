[![Build](https://github.com/trailofbits/vast/actions/workflows/build-test-release.yml/badge.svg)](https://github.com/trailofbits/vast/actions/workflows/build-test-release.yml)

# VAST — Verbose AST

VAST is an experimental frontend for the translation of Clang AST to various MLIR dialects. These dialects allow exploring the codebase at a specific level of abstraction before reaching the low-level LLVM dialect.

## Build

To configure project run `cmake` with following default optaions.
If you want to use system installed `llvm` use:

```
cmake --preset ninja-multi-default -DLLVM_EXTERNAL_LIT=<path lit binary>
```

To use a specific `llvm` provide `-DLLVM_INSTALL_DIR=<llvm instalation path>` option, where `LLVM_INSTALL_DIR` points to directory containing `LLVMConfig.cmake`.


Finally build the project:

```
cmake --build --preset ninja-rel
```

Use `ninja-deb` preset for debug build.

## Run

To run mlir codegen of highlevel dialect use:

```
./builds/ninja-multi-default/bin/vast-cc --from-source <input.c>
```

## Test

```
ctest --preset ninja-deb
```
