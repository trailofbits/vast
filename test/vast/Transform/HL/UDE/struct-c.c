// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK: hl.struct @used
struct used {};

// CHECK: hl.struct @depedent
struct depedent
{
    struct used u;
};

// CHECK-NOT: hl.struct @unused
struct unused {
    struct used u;
};

int main() {
    struct depedent d;
}
