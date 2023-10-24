// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

void foo(int p, char* complicated) {
    // CHECK: hl.switch
    switch (p) {
    // CHECK: hl.case
    // CHECK: hl.const #core.integer<0> : !hl.int
    case 0:
        // CHECK: hl.if
        // CHECK: then
        if (complicated[0] == 'a') {
            // CHECK: hl.if
            // CHECK: then
            if (complicated[1] == 'b') {
    // CHECK: hl.case
    case 1:
                complicated[2] = 'c';
            }
        }
        // CHECK: break
        break;
    }
}
