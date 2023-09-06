// RUN: %vast-front -o %t %s && (%t 1 2 3; test $? -eq 5)

int x = 5;

int main(int argc, char **argv)
{
    return x;
}
