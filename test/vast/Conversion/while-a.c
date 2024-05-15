// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-dce --vast-hl-lower-types --vast-hl-to-ll-cf | %file-check %s

void fn(int a)
{
    // CHECK: ll.scope {
    // CHECK:   ll.br ^bb2
    // CHECK: ^bb1:  // pred: ^bb2
    // CHECK:   ll.br ^bb2
    // CHECK: ^bb2:  // 2 preds: ^bb0, ^bb1
    // CHECK:   ll.cond_scope_ret {{.*}} : i1, ^bb1
    // CHECK: }
    // CHECK: ll.return %0 : none

    while (a)
        --a;
}
