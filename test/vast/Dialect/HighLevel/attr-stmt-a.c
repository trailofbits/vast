// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void f() {
    int x;
    switch (1) {
        case 1:
            x = 1;
// CHECK: hl.attributed_stmt {hl.fallthrough = #hl.fallthrough} : {
// CHECK: hl.null
// CHECK: }
            [[fallthrough]];
        case 2:
            x = 2;
            break;
    }
}
