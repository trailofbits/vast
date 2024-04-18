// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

void float_to_bool() {
    float f;
    bool b = f;
}

// HL: hl.func @_Z13float_to_boolv
// HL:    hl.implicit_cast {{.*}} FloatingToBoolean
// HL: }

// MLIR: llvm.func @_Z13float_to_boolv
// MLIR:    llvm.mlir.constant(0.000000e+00 : f32)
// MLIR:    llvm.fcmp "une"
// MLIR: }

// LLVM: define void @_Z13float_to_boolv
// LLVM:    fcmp une {{.*}} 0.000000e+00
// LLVM: }
