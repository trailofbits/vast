// RUN: %vast-front -o %t %s && %t

int foo(int input) {
    int out = 0;
    do {
        if (input == 12)
            continue;
        if (input == 6)
            return out;

        do {
            if (out > 5)
                break;
            ++out;
        } while (0);

    } while (--input >= 1);

    return out;
}

int main(int argc, char **argv)
{
    if (foo(12) != 5) return 4;
    if (foo(8) != 2) return 5;
    if (foo(4) != 4) return 6;
    if (foo(0) != 1) return 7;

    return 0;
}
