[![Build & Test](https://github.com/trailofbits/vast/actions/workflows/build.yml/badge.svg)](https://github.com/trailofbits/vast/actions/workflows/build.yml)
[![C++ Linter](https://github.com/trailofbits/vast/actions/workflows/linter.yml/badge.svg)](https://github.com/trailofbits/vast/actions/workflows/linter.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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
The pure high-level representation of simple C programs:


<table>
<tr>
<th>
C
</th>
<th>
High-level dialect
</th>
</tr>
<tr>
<td  valign="top">

<pre lang="cpp">
int main() {
    int x = 0;
    int y = x;
    int *z = &x;
}
</pre>
</td>
<td  valign="top">

<pre lang="cpp">
hl.func external @main() -> !hl.int {
    %0 = hl.var "x" : !hl.lvalue<!hl.int> = {
      %4 = hl.const #hl.integer<0> : !hl.int
      hl.value.yield %4 : !hl.int
    }
    %1 = hl.var "y" : !hl.lvalue<!hl.int> = {
      %4 = hl.ref %0 : !hl.lvalue<!hl.int>
      %5 = hl.implicit_cast %4 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
      hl.value.yield %5 : !hl.int
    }
    %2 = hl.var "z" : !hl.lvalue<!hl.ptr<!hl.int>> = {
      %4 = hl.ref %0 : !hl.lvalue<!hl.int>
      %5 = hl.addressof %4 : !hl.lvalue<!hl.int> -> !hl.ptr<!hl.int>
      hl.value.yield %5 : !hl.ptr<!hl.int>
    }
    %3 = hl.const #hl.integer<0> : !hl.int
    hl.return %3 : !hl.int
}
</pre>
</td>
</tr>
<tr>
<td  valign="top">

<pre lang="cpp">
void loop_simple()
{
    for (int i = 0; i < 100; i++) {
        /* ... */
    }
}
</pre>
</td>
<td  valign="top">

<pre lang="cpp">
hl.func external @loop_simple () -> !hl.void {
    %0 = hl.var "i" : !hl.lvalue<!hl.int> = {
      %1 = hl.const #hl.integer<0> : !hl.int
      hl.value.yield %1 : !hl.int
    }
    hl.for {
      %1 = hl.ref %0 : !hl.lvalue<!hl.int>
      %2 = hl.implicit_cast %1 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
      %3 = hl.const #hl.integer<100> : !hl.int
      %4 = hl.cmp slt %2, %3 : !hl.int, !hl.int -> !hl.int
      hl.cond.yield %4 : !hl.int
    } incr {
      %1 = hl.ref %0 : !hl.lvalue<!hl.int>
      %2 = hl.post.inc %1 : !hl.lvalue<!hl.int> -> !hl.int
    } do {
    }
    hl.return
}
</pre>
</td>
</tr>
</table>


For example high-level control flow with standard types:

```
hl.func external  private @loop_simple() -> none {
    %0 = hl.var "i" : i32 = {
      %1 = hl.const #hl.integer<0> : i32
      hl.value.yield %1 : i32
    }
    hl.for {
      %1 = hl.ref %0 : i32
      %2 = hl.implicit_cast %1 LValueToRValue : i32 -> i32
      %3 = hl.const #hl.integer<100> : i32
      %4 = hl.cmp slt %2, %3 : i32, i32 -> i32
      hl.cond.yield %4 : i32
    } incr {
      %1 = hl.ref %0 : i32
      %2 = hl.post.inc %1 : i32 -> i32
    } do {
    }
    hl.return
}
```

Types are lowered according to data-layout embeded into VAST module:

```
  module attributes {
    hl.data.layout = #dlti.dl_spec<
      #dlti.dl_entry<!hl.void, 0 : i32>,
      #dlti.dl_entry<!hl.int, 32 : i32>,
      #dlti.dl_entry<!hl.ptr<!hl.char>, 64 : i32>,
      #dlti.dl_entry<!hl.char, 8 : i32>,
      #dlti.dl_entry<!hl.bool, 1 : i32>
    >
  }
```

## Build

### Dependencies

Currently, it is necessary to use `clang-16` (due to `gcc` bug) to build VAST. On Linux it is also necessary to use `lld` at the moment.

VAST uses `llvm-16` which can be obtained from the [repository](https://apt.llvm.org/) provided by LLVM.

Before building (for Ubuntu) get all the necessary dependencies by running
```
apt-get install build-essential cmake ninja-build libstdc++-12-dev llvm-16 libmlir-16 libmlir-16-dev mlir-16-tools libclang-16-dev
```
or an equivalent command for your operating system of choice.

### Instructions

To configure project run `cmake` with following default options.
In case `clang` isn't your default compiler prefix the command with `CC=clang CXX=clang++`.
If you want to use system installed `llvm` and `mlir` (on Ubuntu) use:

```
cmake --preset ninja-multi-default \
    --toolchain ./cmake/lld.toolchain.cmake \
    -DCMAKE_PREFIX_PATH=/usr/lib/llvm-16/
```

To use a specific `llvm` provide `-DCMAKE_PREFIX_PATH=<llvm & mlir instalation paths>` option, where `CMAKE_PREFIX_PATH` points to directory containing `LLVMConfig.cmake` and `MLIRConfig.cmake`.

Note: vast requires LLVM with RTTI enabled. Use `LLVM_ENABLE_RTTI=ON` if you build your own LLVM.


Finally, build the project:

```
cmake --build --preset ninja-rel
```

Use `ninja-deb` preset for debug build.

## Run

To run mlir codegen of highlevel dialect use.

```
./builds/ninja-multi-default/tools/Release/vast-cc --from-source <input.c>
```

__Note:__ `vast-cc` is to be replaced with `vast-front` a clang-compatible compiler frontend.:


```
./builds/ninja-multi-default/tools/vast-front/Release/vast-front <input.c> -vast-emit-mlir=<dialect> -o <output.mlir>
```

Supported dialects are: `hl`, `ll`, `llvm`

__Disclaimer__ `vast-front` is in early stage of development and may be less durable than `vast-cc`.

## Test

```
ctest --preset ninja-deb
```

## How to Start Extending VAST?

VAST offers a handy script to generate a variety of MLIR primitives. You can find the script at scripts/templater.py. This tool is designed to help you create dialects, passes, operations, types, and attributes interactively.

### Usage

Just run the script. It has been designed to provide a guided process for generating the desired primitives. If you find anything confusing or unintuitive, please don't hesitate to open an issue so that we can address it.

When you run the script, it will generate the basic skeleton for your chosen primitive. It will list the generated files which you can freely modify to provide desired functionality.

## License

VAST is licensed according to the [Apache 2.0](LICENSE) license. VAST links against and uses Clang and LLVM APIs. Clang is also licensed under Apache 2.0, with [LLVM exceptions](https://github.com/llvm/llvm-project/blob/main/clang/LICENSE.TXT).

This research was developed with funding from the Defense Advanced Research Projects Agency (DARPA). The views, opinions and/or findings expressed are those of the author and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.

Distribution Statement A – Approved for Public Release, Distribution Unlimited
