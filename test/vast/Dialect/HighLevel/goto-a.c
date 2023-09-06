// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

int main() {
    // CHECK: [[L:%[0-9]+]] = hl.label.decl "end" : !hl.label

    // CHECK: hl.var "x" : !hl.lvalue<!hl.int>
    int x;

    // CHECK: hl.goto [[L]]
    goto end;

    // CHECK: hl.label [[L]]
    end:;
}
