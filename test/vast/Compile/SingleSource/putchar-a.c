// RUN: %vast-front -o %t %s && %t | %file-check %s
// REQUIRES: abi

int putchar(int);

int main()
{
    // CHECK: a
    putchar('a');
    putchar('\n');
    // Workaround as `vast-front` and `vast-cc` behave differently.
    return 0;
}
