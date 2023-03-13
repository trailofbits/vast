/*
 * Copyright (c) 2021 Trail of Bits, Inc.
 */

#pragma once

#define VAST_COMMON_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wsign-conversion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wconversion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wold-style-cast\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wunused-parameter\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wcast-align\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Woverloaded-virtual\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wctad-maybe-unsupported\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wdouble-promotion\"" )


#define VAST_CLANG_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wambiguous-reversed-operator\"" )

#define VAST_GCC_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wuseless-cast\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wnull-dereference\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wmaybe-uninitialized\"" )

#ifdef __clang__
#define VAST_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  VAST_COMMON_RELAX_WARNINGS \
  VAST_CLANG_RELAX_WARNINGS
#elif __GNUC__
#define VAST_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  VAST_COMMON_RELAX_WARNINGS \
  VAST_GCC_RELAX_WARNINGS
#else
#define VAST_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  VAST_COMMON_RELAX_WARNINGS
#endif

#define VAST_UNRELAX_WARNINGS \
  _Pragma( "GCC diagnostic pop" )

VAST_RELAX_WARNINGS
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Debug.h>
VAST_UNRELAX_WARNINGS

#define DEBUG_TYPE "vast"

namespace vast
{
    #define VAST_REPORT(...) llvm::dbgs() << "[vast] " << llvm::formatv(__VA_ARGS__) << "\n";

    #define VAST_UNREACHABLE(...) \
      VAST_REPORT(__VA_ARGS__) \
      llvm_unreachable(nullptr);

    #define VAST_UNIMPLEMENTED VAST_UNREACHABLE("not implemented: {0}", __PRETTY_FUNCTION__);

    #define VAST_DEBUG(fmt, ...) LLVM_DEBUG(VAST_REPORT(__VA_ARGS__))

    #define VAST_CHECK(cond, fmt, ...) if (!(cond)) { VAST_UNREACHABLE(fmt __VA_OPT__(,) __VA_ARGS__); }

    #define VAST_ASSERT(cond) if (!(cond)) { VAST_UNREACHABLE("assertion: " #cond " failed"); }
}
