// RUN: %vast-front -target x86_64 -c -S -emit-llvm -o %t.vast.ll %s && %cc -target x86_64 -c -S -emit-llvm -xc %s.driver -o %t.clang.ll  && %cc %t.vast.ll %t.clang.ll -o %t && (%t; test $? -eq 0)
// REQUIRES: clang

// Issue #605
// REQUIRES: abi-float-vectors

struct data {
    short a;
    float array[3];
};

float sum(struct data d)
{
    float out = 0.0f;
    out += d.array[0];
    out += d.array[1];
    out += d.array[2];
    return out;
}

int vast_tests() {
    struct data d;
    d.a = 0xffff;
    for (int i = 0; i < 3; ++i)
        d.array[i] = i;

    if (sum(d) != 3.0f)
        return 12;

    d.array[0] = 10.0f;
    if (sum(d) != 13.0f)
        return 13;

    return 0;
}
