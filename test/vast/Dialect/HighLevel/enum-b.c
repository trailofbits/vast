// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

int puts(const char *str);

int main() {
    // CHECK: hl.enum.decl "enum color" : !hl.int<unsigned>  {
    // CHECK:  hl.enum.const "RED" = 0 : si32
    // CHECK:  hl.enum.const "GREEN" = 1 : si32
    // CHECK:  hl.enum.const "BLUE" = 2 : si32
    // CHECK: }

    // CHECK: hl.var "r" : !hl.lvalue<!hl.named_type<"enum color">> =  {
    // CHECK:  [[V1:%[0-9]+]] = hl.enumref "RED" : !hl.int
    // CHECK:  [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] IntegralCast : !hl.int -> !hl.named_type<"enum color">
    // CHECK:  hl.value.yield [[V2]] : !hl.named_type<"enum color">
    // CHECK: }
    enum color { RED, GREEN, BLUE } r = RED;

    // CHECK: hl.switch
    // CHECK:  hl.declref %0 : !hl.lvalue<!hl.named_type<"enum color">>
    switch(r) {
    // CHECK: hl.case
    // CHECK:  hl.enumref "RED" : !hl.int
    case RED:
        puts("red");
        break;
    // CHECK: hl.case
    // CHECK:  hl.enumref "GREEN" : !hl.int
    case GREEN:
        puts("green");
        break;
    // CHECK: hl.case
    // CHECK:  hl.enumref "BLUE" : !hl.int
    case BLUE:
        puts("blue");
        break;
    }
}
