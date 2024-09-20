// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.union @u {hl.transparent_union = #hl.transparent_union}
union u {
    int *a;
    char *b;
} __attribute__ ((transparent_union));

void fun(union u x) {
    return;
}

int main() {
    int x;
// CHECK: hl.call @fun({{.*}}) : (!hl.elaborated<!hl.record<@u>>) -> !hl.void
    fun(&x);
    char y;
// CHECK: hl.compound_literal : !hl.elaborated<!hl.record<@u>>
// CHECK: hl.call @fun({{.*}}) : (!hl.elaborated<!hl.record<@u>>) -> !hl.void
    fun(&y);
    union u z;
// CHECK: hl.call @fun({{.*}}) : (!hl.elaborated<!hl.record<@u>>) -> !hl.void
    fun(z);
}
