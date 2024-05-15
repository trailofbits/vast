// RUN: %vast-front -vast-emit-mlir=llvm -o - %s | %file-check %s

int main() {
    int a = 5;
    // CHECK: {{.* = llvm.load.*}}
    // CHECK-NEXT: {{.* = llvm.mlir.constant.*}}
    // CHECK-NEXT: {{.* = llvm.sub.*}}
    // CHECK-NEXT: {{.*llvm.store.*}}
    // CHECK: {{.* = llvm.load.*}}
    // CHECK-NEXT: {{.*llvm.store.*}}
    int b = -a;
    int c = +a;
    return 0;
}
