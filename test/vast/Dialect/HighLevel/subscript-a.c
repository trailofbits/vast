// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

enum FOO { BAR };

int main(void) {
    int a[1]   = { 0 };
    enum FOO b = BAR;
    // CHECK: hl.subscript {{%[0-9]+}} at [{{%[0-9]+}} : !hl.elaborated<!hl.enum<"FOO">>]
    return a[b];
}
