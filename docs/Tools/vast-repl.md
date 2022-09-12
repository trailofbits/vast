# VAST: REPL

WIP `vast-repl` is an interactive MLIR query and modification tool.

Commands:

```
exit            - exits repl

help            - prints help
load <filename> - loads source from file

show <value>    - displays queried value
    =source         - loaded source code
    =ast            - clang ast
    =module         - current VAST MLIR module
    =symbols        - present symbols in the module

meta <action>   - operates on metadata for given symbol
    =add <symbol> <id> - adds <id> meta to <symbol>
    =get <id>          - gets symbol with <id> meta
```
