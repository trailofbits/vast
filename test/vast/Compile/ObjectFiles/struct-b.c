// RUN: %vast-front -c -vast-pipeline=with-abi -o %t.vast.o %s && %clang -c -xc %s.driver -o %t.clang.o  && %clang %t.vast.o %t.clang.o -o %t && (%t; test $? -eq 0)

struct W_i16
{
    short a;
};

struct Data
{
    int a;
    char b;
    struct W_i16 c;
};

int sum(struct Data d)
{
    return d.a + d.b + d.c.a;
}
