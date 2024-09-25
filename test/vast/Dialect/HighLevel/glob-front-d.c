// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s

// CHECK: hl.var @a, <internal> sc_static
static int a = 2;

int main() {
    int x = a;
    return 0;
}
