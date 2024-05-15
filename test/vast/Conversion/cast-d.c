// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

void int_to_ptr() {
    unsigned long int i;
    void *p = (void*)i;
}

void ptr_to_int() {
    void *p;
    unsigned long int i = (unsigned long int)p;
}

// HL: hl.func @int_to_ptr
// HL:    hl.cstyle_cast {{.*}} IntegralToPointer
// HL: }

// HL: hl.func @ptr_to_int
// HL:    hl.cstyle_cast {{.*}} PointerToIntegral
// HL: }

// MLIR: llvm.func @int_to_ptr
// MLIR:    llvm.inttoptr
// MLIR: }

// MLIR: llvm.func @ptr_to_int
// MLIR:    llvm.ptrtoint
// MLIR: }

// LLVM: define void @int_to_ptr
// LLVM:    inttoptr
// LLVM: }

// LLVM: define void @ptr_to_int
// LLVM:    ptrtoint
// LLVM: }
