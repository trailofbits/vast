// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

void int_to_bool() {
    int i;
    bool b = i;
}

// HL: hl.func @_Z11int_to_boolv
// HL:   hl.var @i : !hl.lvalue<!hl.int>
// HL:   hl.var @b : !hl.lvalue<!hl.bool>
// HL:     hl.implicit_cast {{.*}} LValueToRValue
// HL:     hl.implicit_cast {{.*}} IntegralToBoolean
// HL: }

// MLIR: llvm.func @_Z11int_to_boolv
// MLIR:   [[A:%[0-9]+]] = llvm.alloca {{.*}} x i32
// MLIR:   llvm.alloca {{.*}} x i8
// MLIR:   llvm.mlir.constant(0 : i32)
// MLIR:   llvm.icmp "ne"
// MLIR: }

// LLVM: define void @_Z11int_to_boolv
// LLVM:    %1 = alloca i32
// LLVM:    %2 = alloca i8
// LLVM:    %3 = load i32, ptr %1, align 4
// LLVM:    %4 = icmp ne i32 %3, 0
// LLVM: }
