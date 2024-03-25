// RUN: %vast-front -o %t %s && (%t; test $? -eq 0) \
// RUN:                      && (%t 0 0 0; test $? -eq 42)

int main(int argc, char **argv)
{
    while (argc >= 1) {
        if (argc == 4)
            return 42;
        --argc;
    }
    return argc;
}
