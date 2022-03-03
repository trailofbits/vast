# High Level MLIR CodeGen Example

An example of MLIR project setup. The executable takes the source and generates
a high-level MLIR module for each function declaration in the source AST.

See CMakeLists.txt for the basic configuration of dependencies. It is necessary
to point `cmake` to `vast` and `llvm` installation directory.

## Compilation

```
cmake -DLLVM_INSTALL_DIR=<path to llvm> -DVAST_INSTALL_DIR=<path to vast> -B build -S .

cmake --build build
```

## Usage

```
./build/codegen input.cpp
```
