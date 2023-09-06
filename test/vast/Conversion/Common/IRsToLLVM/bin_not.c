// RUN: %vast-cc --ccopts -xc --from-source %s | %vast-opt --vast-hl-splice-trailing-scopes --vast-hl-lower-types --vast-hl-to-ll-cf --vast-hl-to-ll-vars --vast-irs-to-llvm | FileCheck %s

int main() {
    int a = 5;

    // CHECK: {{.* = llvm.mlir.constant\(-1 : i32.*}}
    // CHECK-NEXT: {{.* = llvm.xor.*}}
    int b = ~a;

    long int c = 5;

    // CHECK: {{.* = llvm.mlir.constant\(-1 : i64.*}}
    // CHECK-NEXT: {{.* = llvm.xor.*}}
    long int d = ~c;

    short e = 5;

    // CHECK: {{.* = llvm.sext.*}}
    // CHECK: {{.* = llvm.mlir.constant\(-1 : i32.*}}
    // CHECK-NEXT: {{.* = llvm.xor.*}}
    short f = ~e;
    return 0;
}
