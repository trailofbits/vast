// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-to-ll-func --vast-fn-args-to-alloca | %file-check %s

int fn(int arg0, int arg1)
{
    // CHECK: %0 = ll.arg_alloca %arg0 : (!hl.int) -> !hl.lvalue<!hl.int>
    // CHECK: %1 = ll.arg_alloca %arg1 : (!hl.int) -> !hl.lvalue<!hl.int>
    // CHECK: {{.*}} = hl.ref %0 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
    // CHECK: {{.*}} = hl.ref %1 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
    return arg0 + arg1;
}
