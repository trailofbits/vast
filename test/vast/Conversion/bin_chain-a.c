// RUN: %vast-front -vast-emit-mlir=llvm %s -o - %s | %file-check %s

int main() {
    int a = 5;
    int b = 5;
    int c = 5;
    // CHECK: ^bb1{{.*}}// pred: ^bb0
    // CHECK: ^bb2{{.*}}// pred: ^bb1
    // CHECK: ^bb3{{.*}}// 2 preds: ^bb1, ^bb2
    // CHECK: ^bb4{{.*}}// 2 preds: ^bb0, ^bb3
    return a && (!b && !c);
}
