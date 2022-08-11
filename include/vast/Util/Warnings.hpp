/*
 * Copyright (c) 2021 Trail of Bits, Inc.
 */

#pragma once

#define VAST_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  _Pragma( "GCC diagnostic ignored \"-Wsign-conversion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wconversion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wold-style-cast\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wunused-parameter\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wcast-align\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wimplicit-int-conversion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wambiguous-reversed-operator\"" )

#define VAST_UNRELAX_WARNINGS \
  _Pragma( "GCC diagnostic pop" )

VAST_RELAX_WARNINGS
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Debug.h>
VAST_UNRELAX_WARNINGS

#define DEBUG_TYPE "vast"

namespace vast
{
    #define VAST_UNREACHABLE(fmt, ...) llvm_unreachable( llvm::formatv(fmt __VA_OPT__(,) __VA_ARGS__).str().c_str() );

    #define VAST_UNIMPLEMENTED VAST_UNREACHABLE("not implemented: {0}", __PRETTY_FUNCTION__);

    #define VAST_DEBUG(fmt, ...) LLVM_DEBUG(llvm::dbgs() << llvm::formatv(fmt, __VA_OPT__(,) __VA_ARGS__));

    #define VAST_CHECK(cond, fmt, ...) if (!(cond)) { VAST_UNREACHABLE(fmt __VA_OPT__(,) __VA_ARGS__); }

    #define VAST_ASSERT(cond) if (!(cond)) { VAST_UNREACHABLE("assertion: " #cond " failed"); }
}
