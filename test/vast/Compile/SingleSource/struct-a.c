// RUN: %vast-front -o %t %s && (%t; test $? -eq 10)

struct data
{
    int a;
    int b;
    int c;
    int d;
    int e;
};

int main(int argc, char **argv)
{
    struct data d = { 0, 1, 2, 3, 4 };
    return d.a + d.b + d.c + d.d + d.e;
}
