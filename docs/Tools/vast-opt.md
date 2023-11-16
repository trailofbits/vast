# VAST: Optimizer

After `mlir` module from `vast-cc` is produced, we can leverage our optimisation pipeline to transform module in various ways. The lowest level we can do is LLVM dialect, from which we can dump LLVM IR if desired.

Overall design philosophy is that passes try to be really modular, self-contained and doing one thing properly - all while trying to preserve provenance metadata. Sometimes this does not exactly hold (transformation from `HL` into `LL` is huge) but it is what we strive for. Passes will more often than not have some dependencies between themselves - always consult documentation if unsure and report an issue if wiki is out of date on this.

### Metadata and passes

**TODO**: Improve once we have examples

**TL;DR**: Vast provided passes always try to keep metadata (and they should do a good job), but for passes from other sources this does not hold and probably some heuristic will be used to re-compute them in best-effort.

## Passes

Passes we have implemented can be roughly grouped into several categories. We also note some of the native mlir passes that are needed to continue with transformations to reach LLVM dialect.

### Type lowering

A common prerequisite for other passes is to lower `HL` types into standard types. This can be done in two steps:
 * `--vast-hl-lower-types`
   - Converts simple (non-struct) types according to provided data layout (embedded in the mlir module metadata).
 * `--vast-hl-structs-to-tuples`
   - Converts `HL` struct types into standard tuples

While these should be commutative, the preferred order is `--vast-hl-lower-types --vast-hl-structs-to-tuples`

### HL -> SCF

 * `--vast-hl-to-scf`
   - Requires:
     + Type lowering
   - Conversion of `HL` control flow ops (currently only `hl.if` and `hl.while`) into their `scf` equivalents. Since `scf` requires `i1` in their conditions, additional casts may be inserted to satisfy this requirement (currently they are emitted in `HL` however this behaviour should customisable eventually.

To produce an LLVM following addition passes must be run
`--convert-scf-to-std --convert-std-to-llvm` and possibly `--vast-hl-to-ll` as well (or some equivalent, depending on how conditions are coerced)

### HL -> LL

* `--vast-hl-to-ll`
  - Requires:
    + Type lowering
    + Some form of control flow lowering
  - Lower all `HL` operation into their LLVM dialect equivalents - this is a rather huge pass, for details see its documentation.

### LLVM Dump

* `--vast-llvm-dump`
  - Requires:
    + Entire module must be in LLVM dialect (or have operation for which conversion hooks are provided)
  - LLVM bitcode is dumped to `llvm::errs()` in human readable form. Since passes can run in parallel, dump to file is non-trivial.

## Example Usage

Let's say we have file `main.c` which we want to lower into some dialect. First let's have a look at some generic invocations we may find handy:

To get mlir module via `vast-cc`
```bash
vast-cc --ccopts -xc --from-source main.c
```
A quick remainder
 * `--ccopts -xc` says we are doing `C` not `C++`
 * `--from-source file` says that source code comes from the file

Once we have the module, we can invoke `vast-opt`, with easiest way being a simple pipe
```bash
vast-cc --ccopts -xc --from-source main.c | vast-opt pass-we-want another-pass-we-want
```
If we want, we can also chain pipes
```bash
vast-cc --ccopts -xc --from-source main.c | vast-opt pass | vast-opt another-pass | ...
```

Now, let's say we want to lower into LLVM bitcode, therefore the invocation will look as follows
```bash
vast-cc --ccopts -xc --from-source main.c | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples
                                                     --vast-hl-to-scf --convert-scf-to-std --convert-std-to-llvm
                                                     --vast-hl-to-ll
```
