// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -
// REQUIRES: compund-literal

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

struct foo {
    int bar;
};

void baz() {
    // compound literal: https://en.cppreference.com/w/c/language/compound_literal
    (struct foo){};

    // these are actually lvalues
    ((struct foo){}).bar = 4;
    &(struct foo){};
}
