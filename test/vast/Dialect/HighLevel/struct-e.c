// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.type.decl @struct.s
// CHECK: hl.record @struct.s : {
// CHECK:  hl.field @a : !hl.int
// CHECK: }
struct s { int a; };

// CHECK: func @f() -> !hl.void
void f() {
    // CHECK: hl.var "v" : !hl.named_type<@struct.s>
    struct s v;
    // CHECK: [[V1:%[0-9]+]] = hl.declref "v" : !hl.named_type<@struct.s>
    // CHECK: [[V2:%[0-9]+]] = hl.member [[V1]] at @a : !hl.named_type<@struct.s> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.constant.int 1 : !hl.int
    // CHECK: hl.assign [[V3]] to [[V2]] : !hl.int
    v.a = 1;
}
