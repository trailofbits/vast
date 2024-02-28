[![Build & Test](https://github.com/trailofbits/vast/actions/workflows/build.yml/badge.svg)](https://github.com/trailofbits/vast/actions/workflows/build.yml)
[![C++ Linter](https://github.com/trailofbits/vast/actions/workflows/linter.yml/badge.svg)](https://github.com/trailofbits/vast/actions/workflows/linter.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# VAST: MLIR for Program Analysis

VAST is a library for program analysis and instrumentation of C/C++ and related
languages. VAST provides a foundation for customizable program representation
for a broad spectrum of analyses. Using the MLIR infrastructure, VAST provides
a toolset to represent C/C++ program at various stages of the compilation and
to transform the representation to the best-fit program abstraction.

For further information check [trailofbits.github.io/vast/](https://trailofbits.github.io/vast/).

## Try VAST

You can experiment with VAST on [compiler explorer](https://godbolt.org/z/3se3q9Tja). Feel free to use VAST and produce MLIR dialects. To specify the desired MLIR output, utilize the `-vast-emit-mlir=<dialect>` option. Currently, the supported options are:

- `-vast-emit-mlir=hl` to generate [high-level](https://trailofbits.github.io/vast/dialects/HighLevel/HighLevel/) dialect.
- `-vast-emit-mlir=llvm` to generate LLVM MLIR dialect.

## License

VAST is licensed according to the [Apache 2.0](LICENSE) license. VAST links against and uses Clang and LLVM APIs. Clang is also licensed under Apache 2.0, with [LLVM exceptions](https://github.com/llvm/llvm-project/blob/main/clang/LICENSE.TXT).

This research was developed with funding from the Defense Advanced Research Projects Agency (DARPA). The views, opinions and/or findings expressed are those of the author and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.

Distribution Statement A – Approved for Public Release, Distribution Unlimited
