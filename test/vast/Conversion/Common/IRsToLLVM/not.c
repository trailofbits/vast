// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-to-ll-cf --vast-hl-to-ll-vars --vast-irs-to-llvm | FileCheck %s

int main() {
    int a = 5;
    // CHECK: {{.* = llvm.icmp "ne".*}}
    // CHECK: {{.* = llvm.xor.*}}
    // CHECK: {{.* = llvm.zext.* to i32.*}}
    int b = !a;
    return 0;
}
