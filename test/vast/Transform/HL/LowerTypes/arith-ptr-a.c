// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types

int main() {
    int *a;
    int *b;
    // CHECK: hl.add {{.*}} : (!hl.ptr<si32>, si32) -> !hl.ptr<si32>
    int *c = a + 1;
    // CHECK: hl.sub {{.*}} : (!hl.ptr<si32>, !hl.ptr<si32>) -> si64
    int  d = a - b;
}
