// RUN: %vast-cc1 -vast-emit-mlir=llvm %s -o - | %file-check %s

int main() {
    float a = 5;
    // CHECK: {{.* = llvm.fcmp "une".*}}
    // CHECK: {{.* = llvm.xor.*}}
    // CHECK: {{.* = llvm.zext.* to i32.*}}
    float b = !a;
    return 0;
}
