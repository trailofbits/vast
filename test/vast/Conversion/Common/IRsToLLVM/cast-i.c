// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

void ptr_to_bool() {
    void *p;
    _Bool b = p;
}

// HL: hl.func @ptr_to_bool
// HL:    hl.implicit_cast {{.*}} PointerToBoolean : !hl.ptr<!hl.void> -> !hl.bool
// HL: }

// MLIR: llvm.func @ptr_to_bool
// MLIR:    llvm.mlir.zero
// MLIR:    llvm.icmp "ne"
// MLIR:    llvm.zext {{.*}} : i1 to i8
// MLIR: }

// LLVM: define void @ptr_to_bool
// LLVM:    icmp ne ptr {{.*}}, null
// LLVM:    zext i1 {{.*}} to i8
// LLVM: }
