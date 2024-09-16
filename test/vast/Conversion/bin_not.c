// RUN: %vast-cc1 -vast-emit-mlir=llvm %s -o - | %file-check %s

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
