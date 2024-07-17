// RUN: %check-hl-to-ll-cf %s | %file-check %s -check-prefix=LL_CF

// LL_CF: ll.scope {
// LL_CF:   ll.br ^bb1
// LL_CF: ^bb1:  // 2 preds: ^bb0, ^bb2
// LL_CF:   ll.cond_scope_ret {{.*}} : i1, ^bb2
// LL_CF: ^bb2:  // pred: ^bb1
// LL_CF:   ll.br ^bb1
// LL_CF: }
// LL_CF: ll.return %0 : none


void fn(int a)
{
    while (a)
        --a;
}
