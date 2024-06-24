// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// This is a function for which clang provides a builtin, which results in some quirky behaviour
// in relation to body generation

// CHECK: hl.return
float rintf (float x) { return x; }

int main (void)
{
// CHECK: hl.call @rintf
    float x = rintf(-1.5);
}
