// RUN: vast-front -o %t %s && (%t; test $? -eq 2)
// RUN: vast-front -o %t %s && (%t 1 2 3; test $? -eq 5)

int main(int argc, char **argv)
{
    int x = 1;
begin:
    x += argc;
end:
    return x;
}
