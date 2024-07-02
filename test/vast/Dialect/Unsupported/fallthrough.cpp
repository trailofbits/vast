// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s
// RUN: %vast-front %s -vast-emit-mlir=hl -o - > %t && %vast-opt %t | diff -B %t -

// CHECK: unsup.stmt "AttributedStmt"
int main(void) {
    int x;

    switch (x) {
        case 0:
            (void)x;
            [[fallthrough]];
        default:
            break;
    }

    return 0;
}
