// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-dce --vast-hl-lower-types --vast-hl-to-ll-cf | %file-check %s

int fn()
{
    // CHECK:   ll.scope {
    // CHECK:     ll.br ^bb2
    // CHECK:   ^bb1:  // pred: ^bb4
    // CHECK:     ll.br ^bb2
    // CHECK:   ^bb2:  // 2 preds: ^bb0, ^bb1
    // CHECK:     ll.cond_scope_ret [[V8:%[0-9]+]] : i1, ^bb3
    // CHECK:   ^bb3:  // pred: ^bb2
    // CHECK:     ll.cond_br [[V13:%[0-9]+]] : i1, ^bb5, ^bb4
    // CHECK:   ^bb4:  // pred: ^bb3
    // CHECK:     ll.br ^bb1
    // CHECK:   ^bb5:  // pred: ^bb3
    // CHECK:     ll.scope_ret
    for ( int i = 0; i < 5; ++i )
        if ( i == 5 )
            break;
        else
            continue;
    // CHECK: }
    return 43;
}
