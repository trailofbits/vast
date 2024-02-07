// RUN: %vast-front -o %t %s && (%t; test $? -eq 1) \
// RUN:                      && (%t 0 0 0; test $? -eq 4)

struct wrapped
{
    int v;
};


int main(int argc, char **argv)
{
    struct wrapped array[] = { { 0 }, { 1 }, { 2 }, { 3 }, { 4 } };
    return array[argc].v;
}
