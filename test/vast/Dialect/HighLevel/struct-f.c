// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

struct s { int i; };

// CHECK: func private @foo(%arg0: !hl.ptr<!hl.named_type<@struct.s>>) -> !hl.int
int foo(struct s *v) {
    // hl.member [[V1:%[0-9]+]] at @i : !hl.ptr<!hl.named_type<@struct.s>> -> !hl.int
    return v->i;
}
