// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @foo external ([[ARG1:%[a-z0-9]+]]: !hl.lvalue<!hl.int>)
int foo();
int foo(int x) {
// CHECK: hl.ref @x
    if (x != 0) {
        return 1;
    }
    return 2;
}

int main() {
// CHECK: hl.call @foo({{%[0-9]+}}) : (!hl.int) -> !hl.int
    foo(3);
}
