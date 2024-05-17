// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

// C_LLVM: llvm.func @fn() {
// C_LLVM:   llvm.return
// C_LLVM: }
void fn() {}
