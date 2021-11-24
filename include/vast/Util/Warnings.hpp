/*
 * Copyright (c) 2021 Trail of Bits, Inc.
 */

#pragma once

#define VAST_RELAX_WARNINGS \
  _Pragma( "clang diagnostic push" ) \
  _Pragma( "clang diagnostic ignored \"-Wsign-conversion\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wconversion\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wold-style-cast\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wunused-parameter\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wcast-align\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wimplicit-int-conversion\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wambiguous-reversed-operator\"" )

#define VAST_UNRELAX_WARNINGS \
  _Pragma( "clang diagnostic pop" )

VAST_RELAX_WARNINGS
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Debug.h>
VAST_UNRELAX_WARNINGS

#define DEBUG_TYPE "vast"

namespace vast
{
    #define UNREACHABLE(fmt, ...) llvm_unreachable( llvm::formatv(fmt __VA_OPT__(,) __VA_ARGS__).str().c_str() );

    #define UNIMPLEMENTED UNREACHABLE("not implemented: {}", __PRETTY_FUNCTION__);

    #define VAST_DEBUG(fmt, ...) LLVM_DEBUG(llvm::dbgs() << llvm::formatv(fmt, __VA_OPT__(,) __VA_ARGS__));

    #define CHECK(cond, fmt, ...) if (!cond) { UNREACHABLE(fmt __VA_OPT__(,) __VA_ARGS__); }
}
