// RUN: vast-cc --ccopts -Wno-gnu-statement-expression --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -Wno-gnu-statement-expression --from-source %s > %t && vast-opt %t | diff -B %t -

int main() {
    // CHECK: hl.alignof.type !hl.int -> !hl.long<unsigned>
    unsigned long si = alignof(int);

    int v;

    // CHECK: hl.var @sv : !hl.long<unsigned>
    // CHECK: hl.alignof.expr -> !hl.long<unsigned>
    // CHECK: hl.declref @v : !hl.int
    unsigned long sv = alignof v;
}
