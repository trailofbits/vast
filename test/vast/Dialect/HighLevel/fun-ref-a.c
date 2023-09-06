// RUN: %vast-front -vast-emit-mlir=hl -o - %s | FileCheck %s

void fun(void);

void fun2(void (*f)(void)){ f(); }

// CHECK: {{.*hl.func .* @fun.*}}
// CHECK: {{.*hl.func .* @main.*}}
int main(void) {
    //CHECK: {{.*hl.funcref @fun.*}}
    fun2(&fun);
    return 0;
}

void fun(void) {}
