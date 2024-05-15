// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-to-ll-func --vast-fn-args-to-alloca | %file-check %s

// ignore external functions
int ext(int);

int fn(int arg0, int arg1)
{
    // CHECK: %0 = ll.arg_alloca %{{.*}} : (!hl.int) -> !hl.lvalue<!hl.int>
    // CHECK-NEXT: {{.*}} = hl.ref %0 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
    return arg0;
}
