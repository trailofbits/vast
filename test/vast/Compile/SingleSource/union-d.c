// RUN: %vast-front -o %t %s && (%t 1 1 1 1; test $? -eq 5)

struct access
{
    int l;
    int h;
};

union data
{
    unsigned long long b;
    struct access s;
};

int main(int argc, char **argv)
{
    union data d;
    d.b = 0xffffffff00000000 + argc;
    d.s.h = 0x12;
    return d.s.l;
}
