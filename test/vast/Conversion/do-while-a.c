// RUN: %check-hl-to-ll-cf %s | %file-check %s -check-prefix=LL_CF
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

// LL_CF:  ll.func @fn external ([[ARG0:%.*]]: !hl.lvalue<si32>) -> none
// LL_CF:    hl.param @a
// LL_CF:    ll.scope {
// LL_CF:       hl.var @sum
// LL_CF:      ll.scope {
// LL_CF:        ll.br ^bb2
// LL_CF:      ^bb1:  // pred: ^bb2
// LL_CF:        [[V8:%[0-9]+]] = hl.ref @a : !hl.lvalue<si32>
// LL_CF:        [[V9:%[0-9]+]] = hl.implicit_cast [[V8]] LValueToRValue : !hl.lvalue<si32> -> si32
// LL_CF:        [[V11:%[0-9]+]] = hl.cmp sgt [[V9]], {{.*}} : si32, si32 -> si32
// LL_CF:        [[V12:%[0-9]+]] = hl.implicit_cast [[V11]] IntegralCast : si32 -> i1
// LL_CF:        ll.cond_scope_ret [[V12]] : i1, ^bb2
// LL_CF:      ^bb2:  // 2 preds: ^bb0, ^bb1
// LL_CF:        [[V6:%[0-9]+]] = hl.ref @a : !hl.lvalue<si32>
// LL_CF:        {{,*}} = hl.pre.dec [[V6]] : !hl.lvalue<si32> -> si32
// LL_CF:        ll.br ^bb1
// LL_CF:      }
// LL_CF:    }

// C_LLVM:  llvm.func @fn(%arg0: i32) {
// C_LLVM:    [[V1:%[0-9]+]] = llvm.alloca {{.*}} : (i64) -> !llvm.ptr
// C_LLVM:    [[V3:%[0-9]+]] = llvm.alloca {{.*}} : (i64) -> !llvm.ptr
// C_LLVM:    llvm.br ^bb1
// C_LLVM:  ^bb1:  // pred: ^bb0
// C_LLVM:    llvm.br ^bb3
// C_LLVM:  ^bb2:  // pred: ^bb3
// C_LLVM:    [[V13:%[0-9]+]] = llvm.icmp "sgt" {{.*}}, {{.*}} : i32
// C_LLVM:    [[V14:%[0-9]+]] = llvm.zext [[V13]] : i1 to i32
// C_LLVM:    [[V15:%[0-9]+]] = llvm.trunc [[V14]] : i32 to i1
// C_LLVM:    llvm.cond_br [[V15]], ^bb3, ^bb4
// C_LLVM:  ^bb3:  // 2 preds: ^bb1, ^bb2
// C_LLVM:    [[V7:%[0-9]+]] = llvm.add {{.*}}, {{.*}}  : i32
// C_LLVM:    llvm.store [[V7]], [[V3]] : i32, !llvm.ptr
// C_LLVM:    [[V10:%[0-9]+]] = llvm.sub {{.*}}, {{.*}}  : i32
// C_LLVM:    llvm.store [[V10]], [[V1]] : i32, !llvm.ptr
// C_LLVM:    llvm.br ^bb2
// C_LLVM:  ^bb4:  // pred: ^bb2
// C_LLVM:    llvm.return
// C_LLVM:  }

void fn(int a)
{
    int sum = 0;
    do {
        sum++;
        --a;
    } while (a > 0);
}
