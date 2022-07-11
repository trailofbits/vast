// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types | FileCheck %s

// CHECK:  hl.struct "X" : {
// CHECK:    hl.field "member_x" : i32
// CHECK:    hl.field "member_y" : i32
// CHECK:  }

struct X
{
    int member_x;
    int member_y;
};

// CHECK-LABEL: func @main() -> i32
int main()
{
    // CHECK: [[V0:%[0-9]+]] = hl.var "var_a" : !hl.lvalue<!hl.named_type<<"X">>>
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[V0]] : !hl.lvalue<!hl.named_type<<"X">>>
    // CHECK: [[V2:%[0-9]+]] = hl.member [[V1]] at "member_x" : !hl.lvalue<!hl.named_type<<"X">>> -> !hl.lvalue<i32>
    // CHECK: [[V3:%[0-9]+]] = hl.constant.int 1 : i32
    // CHECK: [[V4:%[0-9]+]] = hl.assign [[V3]] to [[V2]] : i32
    // CHECK: [[V5:%[0-9]+]] = hl.decl.ref [[V0]] : !hl.lvalue<!hl.named_type<<"X">>>
    // CHECK: [[V6:%[0-9]+]] = hl.member [[V5]] at "member_y" : !hl.lvalue<!hl.named_type<<"X">>> -> !hl.lvalue<i32>
    // CHECK: [[V7:%[0-9]+]] = hl.constant.int 2 : i32
    // CHECK: [[V8:%[0-9]+]] = hl.assign [[V7]] to [[V6]] : i32
    // CHECK: [[V9:%[0-9]+]] = hl.constant.int 0 : i32
    // CHECK: hl.return [[V9]] : i32

    struct X var_a;
    var_a.member_x = 1;
    var_a.member_y = 2;

    return 0;
}
