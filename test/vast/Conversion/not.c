// RUN: %vast-cc1 -vast-emit-mlir=llvm %s -o - | %file-check %s

int main() {
    int a = 5;
    // CHECK: {{.* = llvm.icmp "ne".*}}
    // CHECK: {{.* = llvm.xor.*}}
    // CHECK: {{.* = llvm.zext.* to i32.*}}
    int b = !a;
    return 0;
}
