// RUN: %vast-front -target x86_64 -c -S -emit-llvm -o %t.vast.ll %s && %cc -target x86_64 -c -S -emit-llvm -xc %s.driver -o %t.clang.ll  && %cc %t.vast.ll %t.clang.ll -o %t && (%t; test $? -eq 0)
// REQUIRES: clang

struct data {
    int array[1];
};

int sum(struct data d)
{
    int out = 0;
    out += d.array[0];
    return out;
}

int vast_tests() {
    struct data d;
    d.array[0] = 0;

    if (sum(d) != 0)
        return 11;

    d.array[0] = 12;
    if (sum(d) != 12)
        return 12;

    return 0;
}
