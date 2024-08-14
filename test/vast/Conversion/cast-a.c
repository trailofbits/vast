// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

void ptr_decay_test() {
    int a[2];
    int *p = a;
}

// HL: hl.func @ptr_decay_test {{.*}} () -> !hl.void {
// HL:   hl.var @a
// HL:   hl.var @p
// HL:     hl.implicit_cast {{.*}} ArrayToPointerDecay
// HL: }

// MLIR: llvm.func @ptr_decay_test() {
// MLIR:   [[A:%[0-9]+]] = llvm.alloca {{.*}} !llvm.array<2 x i32> : {{.*}}
// MLIR:   llvm.alloca {{.*}} !llvm.ptr
// MLIR: }

// LLVM: define void @ptr_decay_test()
// LLVM:    %1 = alloca [2 x i32]{{.*}}
// LLVM:    %2 = alloca ptr{{.*}}
// LLVM: }
