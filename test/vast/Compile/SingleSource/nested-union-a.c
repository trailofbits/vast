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
    union data d = { 5 };
    if (d.n.b != 0)
        return 0;
    return d.n.a;
}
