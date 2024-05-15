// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-irs-to-llvm | %file-check %s

int fun(int arg1, int arg2) {
    int res = arg1 && arg2;
    // CHECK: [[LHS:%[0-9]+]] = llvm.load [[V1:%[0-9]+]]
    // CHECK: [[Z:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[LR:%[0-9]+]] = llvm.icmp "ne" [[LHS]], [[Z]] : i32
    // CHECK: llvm.cond_br [[LR]], ^[[TBLOCK:bb[0-9]+]], ^[[RBLOCK:bb[0-9]+]]([[LR]] : i1)
    // CHECK: ^[[TBLOCK]]: // pred: ^[[PRED:bb[0-9]+]]
    // CHECK: [[RHS:%[0-9]+]] = llvm.load [[V2:%[0-9]+]]
    // CHECK: [[RR:%[0-9]+]] = llvm.icmp "ne" [[RHS]], [[Z]]
    // CHECK: llvm.br ^[[RBLOCK]]([[RR]] : i1)
    // CHECK: ^[[RBLOCK]]([[V3:%[0-9]+]]: i1): // 2 preds: ^[[PRED]], ^[[TBLOCK]]
    return res;
}
