// RUN: %vast-front -o %t %s && (%t 1 3 4; test $? -eq 4)
// RUN: %vast-front -o %t %s && (%t; test $? -eq 1)

int main(int argc, char **argv)
{
    return argc;
}
