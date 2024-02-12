// RUN: %vast-front -o %t %s && (%t; test $? -eq 18)

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
    return d.a;
}
