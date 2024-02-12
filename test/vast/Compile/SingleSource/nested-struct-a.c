// RUN: %vast-front -o %t %s && (%t; test $? -eq 5)

struct data
{
    struct nested
    {
        int a;
    } n;
};

int main(int argc, char **argv)
{
    struct data d = { { 5 } };
    return d.n.a;
}
