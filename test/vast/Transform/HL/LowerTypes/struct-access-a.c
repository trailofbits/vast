// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types | %file-check %s

// CHECK:  hl.struct @X : {
// CHECK:    hl.field @member_x : si32
// CHECK:    hl.field @member_y : si32
// CHECK:  }

struct X
{
    int member_x;
    int member_y;
};

int main()
{
    // CHECK: hl.var @var_a : !hl.lvalue<!hl.elaborated<!hl.record<"X">>>
    // CHECK: [[V1:%[0-9]+]] = hl.ref @var_a
    // CHECK: [[V2:%[0-9]+]] = hl.member [[V1]] at @member_x : !hl.lvalue<!hl.elaborated<!hl.record<"X">>> -> !hl.lvalue<si32>
    // CHECK: [[V3:%[0-9]+]] = hl.const #core.integer<1> : si32
    // CHECK: [[V4:%[0-9]+]] = hl.assign [[V3]] to [[V2]] : si32, !hl.lvalue<si32> -> si32
    // CHECK: [[V5:%[0-9]+]] = hl.ref @var_a
    // CHECK: [[V6:%[0-9]+]] = hl.member [[V5]] at @member_y : !hl.lvalue<!hl.elaborated<!hl.record<"X">>> -> !hl.lvalue<si32>
    // CHECK: [[V7:%[0-9]+]] = hl.const #core.integer<2> : si32
    // CHECK: [[V8:%[0-9]+]] = hl.assign [[V7]] to [[V6]] : si32, !hl.lvalue<si32> -> si32
    // CHECK: [[V9:%[0-9]+]] = hl.const #core.integer<0> : si32
    // CHECK: hl.return [[V9]] : si32

    struct X var_a;
    var_a.member_x = 1;
    var_a.member_y = 2;

    return 0;
}
