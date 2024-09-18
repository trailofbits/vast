// RUN: %vast-front -vast-emit-mlir-after=vast-hl-to-ll-cf %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=LL_CF

void fn(int arg)
{
    // LL_CF: ll.scope {
    // LL_CF:   hl.var @a
    // LL_CF:   ll.scope {
    // LL_CF:     ll.cond_br {{.*}} : i1, ^bb1, ^bb2
    // LL_CF:   ^bb1:  // pred: ^bb0
    // LL_CF:     hl.ref @a
    // LL_CF:     hl.ref @arg
    // LL_CF:     hl.assign
    // LL_CF:     ll.br ^bb2
    // LL_CF:   ^bb2:  // 2 preds: ^bb0, ^bb1
    // LL_CF:     ll.scope_ret
    // LL_CF:   }
    // LL_CF: }
    // LL_CF: ll.return {{.*}} : none
    int a = 0;
    {
        if (arg != 0)
            a = arg;
    }
}
