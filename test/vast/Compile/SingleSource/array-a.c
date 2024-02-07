// RUN: %vast-front -o %t %s && (%t; test $? -eq 12) \
// RUN:                      && (%t 0 0 0; test $? -eq 11)

int main(int argc, char **argv)
{
    int array[] = { 10, 12, 13, 14, 11 };
    return array[argc];
}
