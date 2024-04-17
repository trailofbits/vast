// RUN: %vast-front -vast-emit-mlir=llvm %s -o - | %file-check -check-prefix=LLVM %s
// RUN: %vast-front -o %t %s

// LLVM: llvm.func @fn(%arg0: f32, %arg1: f64) {
void fn(float a, double b) {

}

int main(int argc, char **argv) {
    // LLVM: {{.*}}llvm.call @fn{{.*}} : (f32, f64) -> ()
    fn(0.0f, 0.0);
    return 0;
}
