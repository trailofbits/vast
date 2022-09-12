## VAST: Compiler

To compile `c` or `c++` code use `vast-cc` tool:

```
vast-cc --from-source <source.c>
```

The tool compiles `source.c` to clang ast and emits `mlir` in the `high-level` dialect.

To pass additional compiler options use `--ccopts` option.

For further information see `vast-cc --help`.
