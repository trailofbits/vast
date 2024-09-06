// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK: hl.struct @used
struct used { int v; };

int main() {
    struct used u;
    u.v = 0;
}
