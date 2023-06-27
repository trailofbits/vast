// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -
// REQUIRES: libc

#include <stdarg.h>

void format(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);

    va_end(args);
}
