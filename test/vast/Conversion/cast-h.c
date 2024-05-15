// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

void float_to_signed() {
    float f;
    int i = f;
}

void float_to_unsigned() {
    float f;
    unsigned u = f;
}

// HL: hl.func @float_to_signed
// HL:    hl.implicit_cast {{.*}} FloatingToIntegral : !hl.float -> !hl.int
// HL: }

// HL: hl.func @float_to_unsigned
// HL:    hl.implicit_cast {{.*}} FloatingToIntegral : !hl.float -> !hl.int< unsigned >
// HL: }

// MLIR: llvm.func @float_to_signed
// MLIR:    llvm.fptosi {{.*}} : f32 to i32
// MLIR: }

// MLIR: llvm.func @float_to_unsigned
// MLIR:    llvm.fptoui {{.*}} : f32 to i32
// MLIR: }

// LLVM: define void @float_to_signed
// LLVM:    fptosi float {{.*}} to i32
// LLVM: }

// LLVM: define void @float_to_unsigned
// LLVM:    fptoui float {{.*}} to i32
// LLVM: }
