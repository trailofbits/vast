// RUN: vast-front -o %t %s && %t hello | FileCheck %s

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
    // Workaround as `vast-front` and `vast-cc` behave differently.
    return 0;
}
