# VAST: Compiler Driver

`vast-front` serves as the primary `vast` compiler driver for compiling C/C++. It functions as an extension to the Clang compiler and generally supports all Clang's options. Moreover, `vast-front` offers several custom options, primarily designed as points of customization for MLIR generation. All these options are prefixed with `-vast`.

## VAST output targets

- `-vast-emit-mlir=<dialect>`
  - Possible dialects: hl, std, llvm, cir
  - This will execute the translation pipeline up to the specified dialect.

- `-vast-emit-mlir-after=<pass-argument-name>`
  - This will execute the translation pipeline up to the specified mlir pass (including).
  - It uses same names as `opt` to specify passes.

- `-vast-emit-mlir-bytecode` can be used in conjunction with `-vast-emit-mlir=<dialect>` to print the bytecode format instead of the pretty form.

Other available outputs:

- `-vast-emit-llvm`
- `-vast-emit-obj`
- `-vast-emit-asm`

Additional customization options include:

- `-vast-print-pipeline`
- `-vast-disable-<pipeline-step>`
  - Options for `pipeline-step`: "canonicalize", "reduce-hl", "standard-types", etc. (see pipelines section below)

- `-vast-simplify`
  - Simplifies high-level output.

- `-vast-show-locs`
  - Displays locations in MLIR module print.

- `-vast-locs-as-meta-ids`
  - Uses metadata identifiers instead of file locations for locations.

## Debuging and diagnostics

- `-vast-emit-crash-reproducer="reproducer.mlir"`
  - Emits an MLIR transformation crash reproducer; refer to [debugging docs](https://trailofbits.github.io/vast/GettingStarted/debug/).

- `-vast-disable-multithreading`
  - Disables multithreading in pipeline transformations for deterministic debugging.

- `-vast-debug`
  - Prints operations in diagnostics.
  - Prints MLIR stack trace in diagnostics.

- `-vast-disable-vast-verifier`
  - Skips verification of the produced VAST MLIR module.

- `vast-snapshot-at="pass1;...;passN`
  - After each pass that was specified as an option store MLIR into a file (format is `src.pass_name`).
  - `"*"` stores snapshot after every conversion.

## Pipelines

WIP pipelines documentation
