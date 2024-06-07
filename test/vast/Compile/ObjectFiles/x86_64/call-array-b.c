// RUN: %vast-front -target x86_64 -c -S -emit-llvm -o %t.vast.ll %s && %cc -target x86_64 -c -S -emit-llvm -xc %s.driver -o %t.clang.ll  && %cc %t.vast.ll %t.clang.ll -o %t && (%t; test $? -eq 0)

struct data {
    int array[4];
};

int sum(struct data d)
{
    int out = 0;
    for (int i = 0; i < 4; ++i)
        out += d.array[i];
    return out;
}

int vast_tests() {
    struct data d;
    for (int i = 0; i < 4; ++i)
        d.array[i] = i;

    if (sum(d) != 6)
        return 11;

    d.array[0] = 10;
    if (sum(d) != 16)
        return 12;

    return 0;
}
