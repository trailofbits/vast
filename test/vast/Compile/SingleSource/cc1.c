// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -S %s -o %t.s
// RUN: %file-check --input-file=%t.s %s -check-prefix=ASM
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-obj %s -o %t.o
// RUN: objdump -d %t.o | %file-check %s -check-prefix=OBJ
// REQUIRES: data-layout-lowering

void foo() {
    return;
}

//      HL: hl.func @foo () -> !hl.void {
// HL-NEXT:   hl.return
// HL-NEXT: }

//      MLIR: llvm.func @foo() {
// MLIR-NEXT:   llvm.return
// MLIR-NEXT: }

//      LLVM: define void @foo()
// LLVM-NEXT:   ret void
// LLVM-NEXT: }

//      ASM: .globl  foo
// ASM-NEXT: .p2align
// ASM-NEXT: .type foo,@function
// ASM-NEXT: foo:
//      ASM: retq

// OBJ: 0: c3 ret
