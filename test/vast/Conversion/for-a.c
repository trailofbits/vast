// RUN: %check-hl-to-ll-cf %s | %file-check %s -check-prefix=LL_CF

// LL_CF: core.scope {
// LL_CF: core.scope {
// LL_CF-NEXT: ll.br ^bb1

// LL_CF: ^bb1:  // pred: ^bb0
// LL_CF: ll.cond_scope_ret [[V1:%[0-9]+]] : i1, ^bb2

// LL_CF: ^bb2:  // pred: ^bb1
// LL_CF: ll.scope_ret

// LL_CF: }

void fn()
{
    for (int i = 0; i < 15; ++i)
    {
        ++i;
        break;
    }
}
