// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-dce --vast-hl-lower-types --vast-hl-to-ll-cf | %file-check %s

void fn(int a)
{
    // CHECK: ll.scope {
    // CHECK:   ll.br ^bb4
    // CHECK-NEXT: ^bb1:  // pred: ^bb4
    // CHECK:   ll.cond_br {{.*}} : i1, ^bb2, ^bb3
    // CHECK-NEXT: ^bb2:  // pred: ^bb1
    // CHECK:   ll.scope_ret
    // CHECK-NEXT: ^bb3:  // pred: ^bb1
    // CHECK:   ll.br ^bb4
    // CHECK-NEXT: ^bb4:  // 2 preds: ^bb0, ^bb3
    // CHECK:   ll.cond_scope_ret {{.*}} : i1, ^bb1
    // CHECK-NEXT: }
    // CHECK: ll.return %0 : none

    while (a) {
        if (a == 2)
            break;
        --a;
    }
}
