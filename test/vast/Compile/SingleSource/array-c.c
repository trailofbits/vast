// RUN: %vast-front -o %t %s && (%t; test $? -eq 14) \
// RUN:                      && (%t 0 0; test $? -eq 18)


int main(int argc, char **argv)
{
    int array[][2] = { { 1, 11 }, { 2, 12 }, { 3, 13 }, { 4, 14 } };
    return array[argc][0] + array[argc][1];
}
