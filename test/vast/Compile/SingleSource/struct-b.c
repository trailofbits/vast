// RUN: %vast-front -o %t %s && (%t; test $? -eq 14)

struct wrap
{
    int v;
};

struct data
{
    int a;
    struct wrap b;
    struct wrap c;
    struct wrap d;
    int e;
};

int main(int argc, char **argv)
{
    struct data d = { 0, { 1 }, { 2 }, { 3 }, 4 };
    struct wrap w = { 5 };
    d.b = w;
    return d.a + d.b.v + d.c.v + d.d.v + d.e;
}
