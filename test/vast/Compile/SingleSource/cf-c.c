// RUN: %vast-front -o %t %s && %t

int power(int input) {
    int out = 1;
    do {
        --input;
        out *= 2;
    } while (input > 1);

    return out;
}

int main(int argc, char **argv)
{
    if (power(1) != 2) return 1;
    if (power(0) != 2) return 2;
    if (power(10) != 512) return 3;

    return 0;
}
