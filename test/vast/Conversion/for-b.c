// RUN: %check-hl-to-ll-cf %s | %file-check %s -check-prefix=LL_CF

// LL_CF:   core.scope {
// LL_CF:     ll.br ^bb2
// LL_CF:   ^bb1:  // pred: ^bb4
// LL_CF:     ll.br ^bb2
// LL_CF:   ^bb2:  // 2 preds: ^bb0, ^bb1
// LL_CF:     ll.cond_scope_ret [[V8:%[0-9]+]] : i1, ^bb3
// LL_CF:   ^bb3:  // pred: ^bb2
// LL_CF:     ll.cond_br [[V13:%[0-9]+]] : i1, ^bb5, ^bb4
// LL_CF:   ^bb4:  // pred: ^bb3
// LL_CF:     ll.br ^bb1
// LL_CF:   ^bb5:  // pred: ^bb3
// LL_CF:     ll.scope_ret
// LL_CF: }

int fn()
{
    for ( int i = 0; i < 5; ++i )
        if ( i == 5 )
            break;
        else
            continue;
    return 43;
}
