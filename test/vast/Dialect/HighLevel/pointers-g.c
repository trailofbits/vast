// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main() {
    // CHECK: [[N:%[0-9]+]] = hl.var @n : !hl.lvalue<!hl.int>
    // CHECK: hl.var @p : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK:   [[R:%[0-9]+]] = hl.ref [[N]]
    // CHECK:   hl.addressof [[R]] : !hl.lvalue<!hl.int> -> !hl.ptr<!hl.int>
    int n = 1, *p = &n;
    // CHECK: hl.var @pv : !hl.lvalue<!hl.ptr<!hl.void>>
    // CHECK:   BitCast : !hl.ptr<!hl.int> -> !hl.ptr<!hl.void>
    void* pv = p; // int* to void*
    // CHECK: hl.var @p2 : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK:   BitCast : !hl.ptr<!hl.void> -> !hl.ptr<!hl.int>
    int* p2 = pv; // void* to int*
}
