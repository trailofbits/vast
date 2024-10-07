// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s
//
// CHECK: hl.var @ew, <extern_weak>
extern int __attribute__((weak)) ew;
// CHECK: hl.var @wundef, <weak>
int __attribute__((weak)) wundef;
// CHECK: hl.var @wdef, <weak>
int __attribute__((weak)) wdef = 5;
// CHECK: hl.var @ewdef, <weak>
extern int __attribute__((weak)) ewdef = 5;
// CHECK: hl.var @edef, <external>
extern int edef = 5;
// CHECK: hl.var @undef, <external>
int undef;
// CHECK: hl.var @def, <external>
int def = 5;
