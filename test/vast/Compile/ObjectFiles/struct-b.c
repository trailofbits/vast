// RUN: %vast-front -c -vast-pipeline=with-abi -o %t.vast.o %s && %cc -c -xc %s.driver -o %t.clang.o  && %cc %t.vast.o %t.clang.o -o %t && (%t; test $? -eq 0)
// REQUIRES: clang

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

int vast_tests()
{
    struct Data d = { 0, 0, { 0 } };
    int a = sum(d);

    if (a != 0)
        return 1;

    d.a = -200;
    d.b = 5;
    if (sum(d) != -195)
        return 2;

    struct W_i16 nc = { -124 };
    d.c = nc;

    if (sum(d) != -319)
        return 3;

    return 0;
}
