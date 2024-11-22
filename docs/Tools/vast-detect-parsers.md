# VAST: Parser Detection Tool

The VAST parser detection leverages a dialect-based approach, where program data manipulation is abstracted and reduced to a parser dialect. This results in an MLIR representation that combines control-flow constructs with parser-specific operations.

To generate this representation, we provide the `vast-detect-parsers` tool—a customized version of `mlir-opt` that converts VAST dialects into the parser dialect.
To use the tool, simply run:
```bash
vast-detect-parsers -vast-hl-to-parser <input.mlir>
```

Parser conversion can be enhanced with the use of function models, which specify how functions in programs should be interpreted. A default set of models is provided in `Conversion/Parser/default-parsers-config.yaml`. Additional configurations can be supplied via a pass parameter.
