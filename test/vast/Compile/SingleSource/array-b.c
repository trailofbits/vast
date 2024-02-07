// RUN: %vast-front -o %t %s && (%t; test $? -eq 2) \
// RUN:                      && (%t 0 0 0; test $? -eq 5)

int g[] = { 1, 2, 3, 4, 5 };

int main(int argc, char **argv)
{
    return g[argc];
}
