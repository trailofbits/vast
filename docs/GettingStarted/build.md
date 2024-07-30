
## Dependencies

Currently, it is necessary to use `clang` (due to `gcc` bug) to build VAST. On Linux it is also necessary to use `lld` at the moment.

VAST uses `llvm-18` which can be obtained from the [repository](https://apt.llvm.org/) provided by LLVM.

Before building (for Ubuntu) get all the necessary dependencies by running
```
apt-get install build-essential cmake ninja-build libstdc++-12-dev llvm-18 libmlir-18 libmlir-18-dev mlir-18-tools libclang-18-dev
```
or an equivalent command for your operating system of choice.

## Instructions

To configure project run `cmake` with following default options.
In case `clang` isn't your default compiler prefix the command with `CC=clang CXX=clang++`.
If you want to use system installed `llvm` and `mlir` (on Ubuntu) use:

The simplest way is to run

```
cmake --workflow release
```

If this method doesn't work for you, configure the project in the usual way:

```
cmake --preset default
```

To use a specific `llvm` provide `-DCMAKE_PREFIX_PATH=<llvm & mlir instalation paths>` option, where `CMAKE_PREFIX_PATH` points to directory containing `LLVMConfig.cmake` and `MLIRConfig.cmake`.

Note: vast requires LLVM with RTTI enabled. Use `LLVM_ENABLE_RTTI=ON` if you build your own LLVM.


Finally, build the project:

```
cmake --build --preset release
```

Use `debug` preset for debug build.

On Darwin, to make vast able to link binaries with the default sysroot without
having to specify additional flags, use the `VAST_DEFAULT_SYSROOT` CMake flag, e.g.
```
-DVAST_DEFAULT_SYSROOT="$(xcrun --show-sdk-path)"
```

## Run

To run mlir codegen of highlevel dialect use.

```
./builds/default/tools/vast-front/Release/vast-front -vast-emit-mlir=<dialect> <input.c> -o <output.mlir>
```

Supported dialects are: `hl`, `ll`, `llvm`

## Test

```
ctest --preset debug
```
