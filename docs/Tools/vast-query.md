# VAST: Query

`vast-query` is a command line tool to query symbols in the vast generated MLIR. Its primary purpose is to test symbols and their use edges in the produced MLIR. Example of usage:

```
vast-query [options] <input file>
```

Options:

```
  --scope=<function name>      - Show values from scope of a given function
  --show-symbols=<value>       - Show MLIR symbols
    =functions                 -   show function symbols
    =types                     -   show type symbols
    =records                   -   show record symbols
    =vars                      -   show variable symbols
    =globs                     -   show global variable symbols
    =all                       -   show all symbols
  --symbol-users=<symbol name> - Show users of a given symbol
```
