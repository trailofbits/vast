// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

int main() {
    // CHECK: hl.sizeof.type !hl.int -> !hl.long<unsigned>
    unsigned long si = sizeof(int);

    int v;

    // CHECK: [[V:%[0-9]+]] = hl.var "v" : !hl.lvalue<!hl.int>
    // CHECK: [[SV:%[0-9]+]] = hl.var "sv" : !hl.lvalue<!hl.long<unsigned>>
    // CHECK: hl.sizeof.expr -> !hl.long<unsigned>
    // CHECK:  hl.declref [[V]] : !hl.lvalue<!hl.int>
    unsigned long sv = sizeof v;
}
