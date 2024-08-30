// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int puts(const char *str);

int main() {
    // CHECK: hl.enum @color : !hl.int< unsigned >  {
    // CHECK:  hl.enum.const @RED = #core.integer<0> : !hl.int
    // CHECK:  hl.enum.const @GREEN = #core.integer<1> : !hl.int
    // CHECK:  hl.enum.const @BLUE = #core.integer<2> : !hl.int
    // CHECK: }

    // CHECK: [[VAR:%[0-9]+]] = hl.var @r : !hl.lvalue<!hl.elaborated<!hl.enum<"color">>> =  {
    // CHECK:  [[V1:%[0-9]+]] = hl.enumref @RED : !hl.int
    // CHECK:  [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] IntegralCast : !hl.int -> !hl.elaborated<!hl.enum<"color">>
    // CHECK:  hl.value.yield [[V2]] : !hl.elaborated<!hl.enum<"color">>
    // CHECK: }
    enum color { RED, GREEN, BLUE } r = RED;

    // CHECK: hl.switch
    // CHECK:  hl.ref [[VAR]]
    switch(r) {
    // CHECK: hl.case
    // CHECK:  hl.enumref @RED : !hl.int
    case RED:
        puts("red");
        break;
    // CHECK: hl.case
    // CHECK:  hl.enumref @GREEN : !hl.int
    case GREEN:
        puts("green");
        break;
    // CHECK: hl.case
    // CHECK:  hl.enumref @BLUE : !hl.int
    case BLUE:
        puts("blue");
        break;
    }
}
