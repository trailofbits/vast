// RUN: %check-hl-to-ll-cf %s | %file-check %s -check-prefix=LL_CF
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

// LL_CF:  ll.func @fn external ([[ARG0:%.*]]: !hl.lvalue<si32>) -> none {
// LL_CF:    hl.param @a
// LL_CF:    ll.scope {
// LL_CF:      hl.var @sum
// LL_CF:      ll.scope {
// LL_CF:        ll.br ^bb2
// LL_CF:       ^bb1:  // pred: ^bb4
// LL_CF:        [[B1V6:%[0-9]+]] = hl.const #core.integer<0> : si32
// LL_CF:        [[B1V7:%[0-9]+]] = hl.cmp sgt {{.*}}, [[B1V6]] : si32, si32 -> si32
// LL_CF:        [[B1V8:%[0-9]+]] = hl.implicit_cast [[B1V7]] IntegralCast : si32 -> i1
// LL_CF:        ll.cond_scope_ret [[B1V8]] : i1, ^bb2
// LL_CF:      ^bb2:  // 2 preds: ^bb0, ^bb1
// LL_CF:        [[V4:%[0-9]+]] = hl.ref @a : !hl.lvalue<si32>
// LL_CF:        [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<si32> -> si32
// LL_CF:        [[V6:%[0-9]+]] = hl.const #core.integer<42> : si32
// LL_CF:        [[V7:%[0-9]+]] = hl.cmp eq [[V5]], [[V6]] : si32, si32 -> si32
// LL_CF:        [[V8:%[0-9]+]] = hl.implicit_cast [[V7]] IntegralCast : si32 -> i1
// LL_CF:        ll.cond_br [[V8]] : i1, ^bb3, ^bb4
// LL_CF:      ^bb3:  // pred: ^bb2
// LL_CF:        ll.scope_ret
// LL_CF:      ^bb4:  // pred: ^bb2
// LL_CF:        [[V9:%[0-9]+]] = hl.ref @sum : !hl.lvalue<si32>
// LL_CF:        [[V10:%[0-9]+]] = hl.post.inc [[V9]] : !hl.lvalue<si32> -> si32
// LL_CF:        ll.br ^bb1
// LL_CF:      }
// LL_CF:    }

// C_LLVM:  llvm.func @fn({{.*}}: i32) {
// C_LLVM:    [[V1:%[0-9]+]] = llvm.alloca {{.*}} x i32 : (i64) -> !llvm.ptr
// C_LLVM:    [[V3:%[0-9]+]] = llvm.alloca {{.*}} x i32 : (i64) -> !llvm.ptr
// C_LLVM:    llvm.br ^bb1
// C_LLVM:  ^bb1:  // pred: ^bb0
// C_LLVM:    llvm.br ^bb3
// C_LLVM:  ^bb2:  // pred: ^bb5
// C_LLVM:    [[B2V6:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// C_LLVM:    {{.*}} = llvm.icmp "sgt" {{.*}}, [[B2V6]] : i32
// C_LLVM:    llvm.cond_br {{.*}}, ^bb3, ^bb6
// C_LLVM:  ^bb3:  // 2 preds: ^bb1, ^bb2
// C_LLVM:    {{.*}} = llvm.mlir.constant(42 : i32) : i32
// C_LLVM:    llvm.cond_br {{.*}}, ^bb4, ^bb5
// C_LLVM:  ^bb4:  // pred: ^bb3
// C_LLVM:    llvm.br ^bb6
// C_LLVM:  ^bb5:  // pred: ^bb3
// C_LLVM:    [[V10:%[0-9]+]] = llvm.load [[V3]] : !llvm.ptr -> i32
// C_LLVM:    [[V11:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
// C_LLVM:    [[V12:%[0-9]+]] = llvm.add [[V10]], [[V11]]  : i32
// C_LLVM:    llvm.br ^bb2
// C_LLVM:  ^bb6:  // 2 preds: ^bb2, ^bb4
// C_LLVM:    llvm.return
// C_LLVM:  }

void fn(int a)
{
    int sum = 1;
    do {
        if (a == 42)
            break;
        sum++;
        continue;
    } while (a > 0);
}
