// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-irs-to-llvm | %file-check %s

unsigned long sizeof_int() {
    // CHECK: llvm.mlir.constant(4 : i64) : i64
    return sizeof(int);
}
