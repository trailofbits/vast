[![Build & Test](https://github.com/trailofbits/vast/actions/workflows/build.yml/badge.svg)](https://github.com/trailofbits/vast/actions/workflows/build.yml)
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

## License

VAST is licensed according to the [Apache 2.0](LICENSE) license. VAST links against and uses Clang and LLVM APIs. Clang is also licensed under Apache 2.0, with [LLVM exceptions](https://github.com/llvm/llvm-project/blob/main/clang/LICENSE.TXT).

This research was developed with funding from the Defense Advanced Research Projects Agency (DARPA). The views, opinions and/or findings expressed are those of the author and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.
 
Distribution Statement A – Approved for Public Release, Distribution Unlimited
