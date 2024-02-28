# VAST: MLIR for Program Analysis

VAST is a library for program analysis and instrumentation of C/C++ and related
languages. VAST provides a foundation for customizable program representation
for a broad spectrum of analyses. Using the MLIR infrastructure, VAST provides
a toolset to represent C/C++ program at various stages of the compilation and
to transform the representation to the best-fit program abstraction.

Whether static or dynamic, program analysis often requires a specific view
of the source code. The usual requirements for a representation is to be
easily analyzable, i.e., have a reasonably small set of operations, be truthful
to the semantics of the analyzed program, and the analysis must be relatable to
the source. It is also beneficial to access the source at various abstraction
levels.

The current state-of-the-art tools leverage compiler infrastructures to perform
program analysis. This approach is beneficial because it remains truthful to the
executed program semantics, whether AST or LLVM IR. However, these
representations come at a cost as they are designed for optimization and code
generation, rather than for program analysis.

The Clang AST is unoptimized and too complex for interpretation-based analysis.
Also, it lacks program features that Clang inserts during its LLVM code
generation process. On the other hand, LLVM is often too low-level and hard to
relate to high-level program constructs.

VAST is a new compiler front/middle-end designed for program analysis. It
transforms parsed C and C++ code, in the form of Clang ASTs, into a high-level
MLIR dialect. The high level dialect is then progressively lowered all the way
down to LLVM IR. This progression enables VAST to represent the code as a tower
of IRs in multiple MLIR dialects. The MLIR allows us to capture high-level
features from AST and interleave them with low-level dialects.

## Try VAST

You can experiment with VAST on [compiler
explorer](https://godbolt.org/z/3se3q9Tja). Feel free to use VAST and produce
MLIR dialects. To specify the desired MLIR output, utilize the
`-vast-emit-mlir=<dialect>` option. Currently, the supported options are:

- `-vast-emit-mlir=hl` to generate
[high-level](https://trailofbits.github.io/vast/dialects/HighLevel/HighLevel/)
dialect.  - `-vast-emit-mlir=llvm` to generate LLVM MLIR dialect.

Refer to the [vast-front
documentation](https://trailofbits.github.io/vast/Tools/vast-front/) for
additional details.

## A Tower of IRs

The feature that differentiates our approach is that the program representation
can hold multiple representations simultaneously, the so-called `tower of IRs`.
One can imagine the tower as multiple MLIR modules side-by-side in various
dialects. Each layer of the tower represents a specific stage of compilation. At
the top is a high-level dialect relatable to AST, and at the bottom is a
low-level LLVM-like dialect. Layers are interlinked with location information.
Higher layers can also be seen as metadata for lower layers.

This feature simplifies analysis built on top of VAST IR in multiple ways. It
naturally provides __provenance__ to higher levels dialects (and source code)
from the low levels. Similarly, one can reach for low-level representation from
the high-level source view. This can have multiple utilizations.  One of them is
relating analysis results to the source. For a user, it is invaluable to
represent results in the language of what they see, that is, the high-level
representation of the source. For example, using provenance, one can link the
values in low-level registers to variable names in the source.  Furthermore,
this streamlines communication from the user to the analysis backend and back in
the interactive tools and also allows the automatic analysis to query the
best-fit representation at any time.

The provenance is invaluable for static analysis too. It is often advantageous
to perform analysis as an abstract interpretation of the low-level
representation and relate it to high-level constructs. For example, when trying
to infer properties about control flow, like loop invariants, one can examine
high-level operations and relate the results to low-level analysis using
provenance links.

We expect to provide a DSL library for design of custom program representation
abstraction on top of our tower of IRs. The library will provide utilities to
link other dialects to the rest of the tower so that the provenance is usable
outside the main pipeline.

## Dialects

As a foundation, VAST provides backbone dialects for the tower of IRs.
A high-level dialect `hl` is a faithful representation of Clang AST. While
intermediate dialects represent compilation artifacts like ABI lowering of macro
expansions. Whenever it is possible, we try to utilize standard dialects. At the
bottom of the tower, we have the `llvm` dialect. For features that are not
present in the `llvm` dialect, we utilize our low-level dialect `ll`. We
leverage a `meta` dialect to provide provenance utilities. The currently
supported features are documented in automatically generated dialect
[docs](https://github.com/trailofbits/vast/tree/master/docs).

For types, we provide high-level types from Clang AST enriched by value
categories. This allows referencing types as presented in the source. In the
rest of the tower, we utilize standard or llvm types, respectively.

One does not need to utilize the tower of IRs but can craft a specific
representation that interleaves multiple abstractions simultaneously.
