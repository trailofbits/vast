// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -
// REQUIRES: static_assert

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

_Static_assert(sizeof(struct flex) == sizeof(int), "");
// sizeof(f) does not include the size of statically-declared elements
_Static_assert(sizeof(f) == sizeof(struct flex), "");

// this only builds because .elems is not initialized:
struct flex g[2];
