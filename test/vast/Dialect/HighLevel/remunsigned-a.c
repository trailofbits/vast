// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

void div(unsigned int arg1, unsigned int arg2) {

    // CHECK: [[R:%[0-9]+]] = hl.urem [[V1:%[0-9]+]], [[V2:%[0-9]+]] : (!hl.int< unsigned >, !hl.int< unsigned >) -> !hl.int< unsigned >
    unsigned int res = arg1 % arg2;
}
