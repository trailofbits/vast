// RUN: %vast-front -target x86_64 -c -S -emit-llvm -o %t.vast.ll %s && %cc -target x86_64 -c -S -emit-llvm -xc %s.driver -o %t.clang.ll  && %cc %t.vast.ll %t.clang.ll -o %t && (%t; test $? -eq 0)

struct data {
    float array[1];
};

float sum(struct data d)
{
    float out = 0.0f;
    out += d.array[0];
    return out;
}

int vast_tests() {
    struct data d;
    d.array[0] = 0.0f;

    if (sum(d) != 0.0f)
        return 11;

    d.array[0] = 10.0f;
    if (sum(d) != 10.0f)
        return 12;

    return 0;
}
