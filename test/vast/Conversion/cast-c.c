// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

void float_to_double_cast() {
    float f;
    double d = f;
}

void double_to_float_cast() {
    double d;
    float f = d;
}

// HL: hl.func @float_to_double_cast
// HL:    hl.implicit_cast {{.*}} FloatingCast : !hl.float -> !hl.double
// HL: }

// HL: hl.func @double_to_float_cast
// HL:    hl.implicit_cast {{.*}} FloatingCast : !hl.double -> !hl.float
// HL: }

// MLIR: llvm.func @float_to_double_cast
// MLIR:    llvm.fpext {{.*}} : f32 to f64
// MLIR: }

// MLIR: llvm.func @double_to_float_cast
// MLIR:    llvm.fptrunc {{.*}} : f64 to f32
// MLIR: }

// LLVM: define void @float_to_double_cast
// LLVM:    fpext float {{.*}} to double
// LLVM: }

// LLVM: define void @double_to_float_cast
// LLVM:    fptrunc double {{.*}} to float
// LLVM: }
