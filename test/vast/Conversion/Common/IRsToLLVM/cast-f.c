// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

void float_to_bool() {
    float f;
    _Bool b = f;
}

// HL: hl.func @float_to_bool
// HL:    hl.implicit_cast {{.*}} FloatingToBoolean
// HL: }

// MLIR: llvm.func @float_to_bool
// MLIR:    llvm.mlir.constant(0.000000e+00 : f32)
// MLIR:    llvm.fcmp "une"
// MLIR:    llvm.zext {{.*}} : i1 to i8
// MLIR: }

// LLVM: define void @float_to_bool
// LLVM:    fcmp une {{.*}} 0.000000e+00
// LLVM: }
