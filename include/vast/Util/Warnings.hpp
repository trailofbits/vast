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
  _Pragma( "GCC diagnostic ignored \"-Wdouble-promotion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wshadow\"")

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

#ifdef VAST_ENABLE_EXCEPTIONS
#include <stdexcept>
#include <sstream>
#endif

#define SECOND_ARG(A,B,...) B
#define CONCAT(A,B) A ## B

#define DETECT_EXIST_TRUE ~,1

// DETECT_EXIST merely concats a converted macro to the end of DETECT_EXIST_TRUE.
// If empty, DETECT_EXIST_TRUE converts fine.  If not 0 remains second argument.
#define DETECT_EXIST(X) DETECT_EXIST_IMPL(CONCAT(DETECT_EXIST_TRUE,X), 0, ~)
#define DETECT_EXIST_IMPL(...) SECOND_ARG(__VA_ARGS__)

#define DEBUG_TYPE "vast"

namespace vast {

#ifdef VAST_ENABLE_EXCEPTIONS
    struct error_throwing_stream {
        explicit error_throwing_stream() : ss(buff) {}

        ~error_throwing_stream() noexcept(false) {
            ss.flush();
            throw std::runtime_error(buff);
        }

        template< typename T >
        error_throwing_stream& operator<<(T &&value) {
            ss << value;
            return *this;
        }

      private:
        std::string buff;
        llvm::raw_string_ostream ss;
    };

    error_throwing_stream& vast_error();
#else
    llvm::raw_ostream& vast_error();
#endif
    llvm::raw_ostream& vast_debug();

    #define VAST_ERROR(...) do { \
      vast_error() << "[VAST Error] " << llvm::formatv(__VA_ARGS__) << "\n"; \
    } while(0)

    #define VAST_REPORT(...) do { \
      vast_debug() << "[VAST Debug] " << llvm::formatv(__VA_ARGS__) << "\n"; \
    } while(0)

    #define VAST_UNREACHABLE(...) do { \
      VAST_ERROR(__VA_ARGS__); \
      llvm_unreachable(nullptr); \
    } while (0)

    #define VAST_UNIMPLEMENTED VAST_UNREACHABLE("not implemented: {0}", __PRETTY_FUNCTION__)

    #define VAST_UNIMPLEMENTED_MSG(msg) \
      VAST_UNREACHABLE("not implemented: {0} because {1}", __PRETTY_FUNCTION__, msg)

    #define VAST_UNIMPLEMENTED_IF(cond) \
      if (cond) { VAST_UNREACHABLE("not implemented: {0}", __PRETTY_FUNCTION__); }

    #define VAST_DEBUG(fmt, ...) LLVM_DEBUG(VAST_REPORT(__VA_ARGS__))

    #define VAST_CHECK(cond, fmt, ...) if (!(cond)) { VAST_UNREACHABLE(fmt __VA_OPT__(,) __VA_ARGS__); }

    #define VAST_ASSERT(cond) if (!(cond)) { VAST_UNREACHABLE("assertion: " #cond " failed"); }

    #define VAST_TODO(fmt, ... ) VAST_UNREACHABLE("[vast-todo]: " # fmt __VA_OPT__(,) __VA_ARGS__ )

} // namespace vast
