// RUN: %vast-front -vast-emit-mlir=llvm %s -o - | %file-check -check-prefix=LLVM %s
// RUN: %vast-front -o %t %s

// LLVM: llvm.func @fadd(%arg0: f32, %arg1: f32) -> f32 {
float fadd(float a, float b) {
    return a + b;
}

// LLVM: llvm.func @dadd(%arg0: f64, %arg1: f64) -> f64 {
double dadd(double a, double b) {
    return a + b;
}

int main(int argc, char **argv) {
    // LLVM: {{.*}} = llvm.call @fadd{{.*}} : (f32, f32) -> f32
    float fa = fadd(0.1f, 0.2f);
    // LLVM: {{.*}} llvm.call @dadd{{.*}} : (f64, f64) -> f64
    double da = dadd(0.1, 0.2);
    return 0;
}
