// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int f(int), fc(const int);

int main() {
    // CHECK: hl.var "pc" : !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int< const >>) -> (!hl.int)>>>>
    // CHECK:   hl.funcref @f : !core.fn<(!hl.lvalue<!hl.int>) -> (!hl.int)>
    int (*pc)(const int) = f;
    // CHECK: hl.var "p" : !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int>) -> (!hl.int)>>>>
    // CHECK: hl.funcref @fc : !core.fn<(!hl.lvalue<!hl.int< const >>) -> (!hl.int)>
    int (*p)(int) = fc;
    // CHECK: hl.assign [[P:%[0-9]+]] to [[PC:%[0-9]+]] : !hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int>) -> (!hl.int)>>>, !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int< const >>) -> (!hl.int)>>>> -> !hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int< const >>) -> (!hl.int)>>>
    pc = p;
}
