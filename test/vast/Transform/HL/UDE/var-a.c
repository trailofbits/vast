// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK: hl.var @a
int a;

// CHECK-NOT: hl.var @e
extern int e;

// CHECK: hl.var @s
static int s;

void local(void) {
    // CHECK: hl.var @l
    int l;
}