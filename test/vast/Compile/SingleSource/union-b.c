// RUN: %vast-front -o %t %s && (%t 1 1 1 1; test $? -eq 5)

union data
{
    int a;
    unsigned long long b;
    char c;
};

int main(int argc, char **argv)
{
    union data d;
    d.b = 0xffffffff00000012;
    d.a = argc;
    return d.c;
}
