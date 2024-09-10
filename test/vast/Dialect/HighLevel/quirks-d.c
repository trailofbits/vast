// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

struct flex {
    int count;
    int elems[]; // <-- flexible array member
};

// this lays out the object exactly as expected
struct flex f = {
    .count = 3,
    .elems = {32, 31, 30}
};

// CHECK: hl.static_assert failed : false
_Static_assert(sizeof(struct flex) == sizeof(int), "");
// sizeof(f) does not include the size of statically-declared elements
// CHECK: hl.static_assert failed : false
_Static_assert(sizeof(f) == sizeof(struct flex), "");

// this only builds because .elems is not initialized:
struct flex g[2];
