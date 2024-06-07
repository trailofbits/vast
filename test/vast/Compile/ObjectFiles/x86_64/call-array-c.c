// RUN: %vast-front -target x86_64 -c -S -emit-llvm -o %t.vast.ll %s && %cc -target x86_64 -c -S -emit-llvm -xc %s.driver -o %t.clang.ll  && %cc %t.vast.ll %t.clang.ll -o %t && (%t; test $? -eq 0)

struct data {
    short a;
    int array[3];
};

int sum(struct data d)
{
    int out = 0;
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

    if (sum(d) != 3)
        return 12;

    d.array[0] = 10;
    if (sum(d) != 13)
        return 13;

    return 0;
}
