// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

#include <stdarg.h>

// CHECK: hl.func @__builtin_va_end (!hl.lvalue<!hl.ptr<!hl.record<"__va_list_tag">>>) -> !hl.void
// CHECK: hl.func @__builtin_va_start (!hl.lvalue<!hl.ptr<!hl.record<"__va_list_tag">>>, ...) -> !hl.void
// CHECK: hl.typedef "va_list" : !hl.elaborated<!hl.typedef<"__builtin_va_list">>

// CHECK: hl.func @format ({{%.*}}: !hl.lvalue<!hl.ptr<!hl.char< const >>>, ...) -> !hl.void
void format(const char *fmt, ...) {
    // CHECK: hl.var "args" : !hl.lvalue<!hl.elaborated<!hl.typedef<"va_list">>>
    va_list args;
    // CHECK: hl.call @__builtin_va_start
    va_start(args, fmt);
    // CHECK: hl.call @__builtin_va_end
    va_end(args);
}
