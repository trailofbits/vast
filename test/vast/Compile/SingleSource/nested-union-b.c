// RUN: %vast-front -o %t %s && (%t; test $? -eq 5)

union data
{
    unsigned long long l;
    struct P
    {
        int a;
        int b;
    } n;
};

int main(int argc, char **argv)
{
    union data d;
    struct P p = { 1, 2 };
    d.n = p;
    if (d.l != 0x0000000200000001)
        return 128;
    return 5;
}
