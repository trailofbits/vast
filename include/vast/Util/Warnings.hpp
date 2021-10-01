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
  _Pragma( "clang diagnostic ignored \"-Wimplicit-int-conversion\"" )

#define VAST_UNRELAX_WARNINGS \
  _Pragma( "clang diagnostic pop" )
