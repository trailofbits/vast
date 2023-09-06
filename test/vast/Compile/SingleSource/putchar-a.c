// RUN: %vast-front -o %t %s && %t | FileCheck %s

int putchar(int);

int main()
{
    // CHECK: a
    putchar('a');
    putchar('\n');
    // Workaround as `vast-front` and `vast-cc` behave differently.
    return 0;
}
