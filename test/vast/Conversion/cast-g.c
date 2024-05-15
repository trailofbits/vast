// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

void bool_to_int() {
    _Bool b;
    int i = b;
}

// HL: hl.func @bool_to_int
// HL:    hl.implicit_cast {{.*}} IntegralCast : !hl.bool -> !hl.int
// HL: }

// MLIR: llvm.func @bool_to_int
// MLIR:    llvm.zext {{.*}} : i8 to i32
// MLIR: }

// LLVM: define void @bool_to_int
// LLVM:    zext i8 {{.*}} to i32
// LLVM: }
