// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

int puts(const char *str);

int main() {
    // CHECK: hl.enum.decl @color : !hl.int<unsigned>
    // CHECK:  hl.enum.const @RED = 0 : si32
    // CHECK:  hl.enum.const @GREEN = 1 : si32
    // CHECK:  hl.enum.const @BLUE = 2 : si32
    // CHECK: hl.var @r : !hl.named_type<@color>
    enum color { RED, GREEN, BLUE } r = RED;

    // CHECK: hl.switch cond
    // CHECK:  hl.declref @r : !hl.named_type<@color>
    switch(r) {
    // CHECK: hl.case
    // CHECK:  hl.declref @RED : !hl.int
    case RED:
        puts("red");
        break;
    // CHECK: hl.case
    // CHECK:  hl.declref @GREEN : !hl.int
    case GREEN:
        puts("green");
        break;
    // CHECK: hl.case
    // CHECK:  hl.declref @BLUE : !hl.int
    case BLUE:
        puts("blue");
        break;
    }
}
