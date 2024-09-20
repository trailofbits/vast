// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %vast-opt --vast-hl-to-hl-builtin | %file-check %s

#include <stdarg.h>

// CHECK: hl.func @__builtin_va_end
// CHECK: hl.func @__builtin_va_start
// CHECK: hl.typedef @va_list

// CHECK: hl.func @format {{.*}} ({{%.*}}: !hl.lvalue<!hl.ptr<!hl.char< const >>>, ...) -> !hl.void
void format(const char *fmt, ...) {
    // CHECK: hl.var @args : !hl.lvalue<!hl.elaborated<!hl.typedef<@va_list>>>
    va_list args;
    // CHECK: hlbi.va_start
    va_start(args, fmt);
    // CHECK: hlbi.va_end
    va_end(args);
}
