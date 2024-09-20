// RUN: %check-hl-to-ll-cf %s | %file-check %s -check-prefix=LL_CF
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

// LL_CF: core.scope {
// LL_CF:   ll.br ^bb1
// LL_CF: ^bb1:  // 2 preds: ^bb0, ^bb2
// LL_CF:   [[W1:%[0-9]+]] = hl.const #core.integer<0> : si32
// LL_CF:   [[W2:%[0-9]+]] = hl.implicit_cast [[W1]] IntegralCast : si32 -> i1
// LL_CF:   ll.cond_scope_ret [[W2]] : i1, ^bb2
// LL_CF: ^bb2:  // pred: ^bb1
// LL_CF:   ll.br ^bb1
// LL_CF: }

// C_LLVM:    llvm.br ^bb1
// C_LLVM:  ^bb1:  // pred: ^bb0
// C_LLVM:    llvm.br ^bb2
// C_LLVM:  ^bb2:  // 2 preds: ^bb1, ^bb3
// C_LLVM:    [[W1:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// C_LLVM:    [[W2:%[0-9]+]] = llvm.trunc [[W1]] : i32 to i1
// C_LLVM:    llvm.cond_br [[W2]], ^bb3, ^bb4
// C_LLVM:  ^bb3:  // pred: ^bb2
// C_LLVM:    llvm.br ^bb2
// C_LLVM:  ^bb4:  // pred: ^bb2
// C_LLVM:    llvm.return

void fn_while() {
    while (0) {}
}

// LL_CF: core.scope {
// LL_CF:   ll.br ^bb2
// LL_CF: ^bb1:  // pred: ^bb2
// LL_CF:   [[DW1:%[0-9]+]] = hl.const #core.integer<0> : si32
// LL_CF:   [[DW2:%[0-9]+]] = hl.implicit_cast [[DW1]] IntegralCast : si32 -> i1
// LL_CF:   ll.cond_scope_ret [[DW2]] : i1, ^bb2
// LL_CF: ^bb2:  // 2 preds: ^bb0, ^bb1
// LL_CF:   ll.br ^bb1
// LL_CF: }

// C_LLVM:    llvm.br ^bb1
// C_LLVM:  ^bb1:  // pred: ^bb0
// C_LLVM:    llvm.br ^bb3
// C_LLVM:  ^bb2:  // pred: ^bb3
// C_LLVM:    [[DW1:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// C_LLVM:    [[DW2:%[0-9]+]] = llvm.trunc [[DW1]] : i32 to i1
// C_LLVM:    llvm.cond_br [[DW2]], ^bb3, ^bb4
// C_LLVM:  ^bb3:  // 2 preds: ^bb1, ^bb2
// C_LLVM:    llvm.br ^bb2
// C_LLVM:  ^bb4:  // pred: ^bb2
// C_LLVM:    llvm.return

void fn_do_while() {
    do {} while(0);
}

// LL_CF: core.scope {
// LL_CF:   ll.br ^bb2
// LL_CF: ^bb1:  // pred: ^bb3
// LL_CF:   ll.br ^bb2
// LL_CF: ^bb2:  // 2 preds: ^bb0, ^bb1
// LL_CF:   [[F1:%[0-9]+]] = hl.const #core.integer<0> : si32
// LL_CF:   [[F2:%[0-9]+]] = hl.implicit_cast [[F1]] IntegralCast : si32 -> i1
// LL_CF:   ll.cond_scope_ret [[F2]] : i1, ^bb3
// LL_CF: ^bb3:  // pred: ^bb2
// LL_CF:   ll.br ^bb1
// LL_CF: }

// C_LLVM:    llvm.br ^bb1
// C_LLVM:  ^bb1:  // pred: ^bb0
// C_LLVM:    llvm.br ^bb3
// C_LLVM:  ^bb2:  // pred: ^bb4
// C_LLVM:    llvm.br ^bb3
// C_LLVM:  ^bb3:  // 2 preds: ^bb1, ^bb2
// C_LLVM:    [[F1:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// C_LLVM:    [[F2:%[0-9]+]] = llvm.trunc [[F1]] : i32 to i1
// C_LLVM:    llvm.cond_br [[F2]], ^bb4, ^bb5
// C_LLVM:  ^bb4:  // pred: ^bb3
// C_LLVM:    llvm.br ^bb2
// C_LLVM:  ^bb5:  // pred: ^bb3
// C_LLVM:    llvm.return

void fn_for() {
    for (;0;) {}
}
