# VAST: Language Server Protocol

VAST provides an implementation of LSP language server in the form of the `vast-lsp-server` tool. This tool interacts with the MLIR C++ API to support rich language queries, such as “Find Definition”.

The tool easily integrates with [VSCode](https://code.visualstudio.com/) extension [MLIR](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir). The user needs to point the extension to `mlir-lsp-server`. To do so, one can create a symbolic link named `mlir-lsp-server` to point to built `vast-lsp-server`.

## Build

To build `vast-lsp-server` use:

```
cmake --build <build-dir> --target vast-lsp-server
```
