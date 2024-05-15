// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-dce --vast-hl-lower-types --vast-hl-to-ll-cf | %file-check %s

void fn()
{
    // CHECK: ll.scope {
    // CHECK: ll.scope {
    // CHECK-NEXT: ll.br ^bb1

    // CHECK: ^bb1:  // pred: ^bb0
    // CHECK: ll.cond_scope_ret [[V1:%[0-9]+]] : i1, ^bb2

    // CHECK: ^bb2:  // pred: ^bb1
    // CHECK: ll.scope_ret

    // CHECK: }
    for (int i = 0; i < 15; ++i)
    {
        ++i;
        break;
    }
}
