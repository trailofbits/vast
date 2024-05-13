// RUN: %vast-front -o %t %s && %t hello | %file-check %s
// REQUIRES: ssa-core-scope

int puts(const char *);

int main(int argc, char **argv)
{
    if (argc < 1)
    {
        puts("Nothing");
        return 0;
    }

    // CHECK: hello
    puts(argv[1]);
}
