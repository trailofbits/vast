// RUN: %vast-front -o %t -vast-output-sarif=%t.sarif %s && test -f %t.sarif && cat %t.sarif | %file-check %s
// REQUIRES: sarif

// CHECK: "informationUri": "https://github.com/trailofbits/vast.git",
// CHECK: "name": "vast-front",
// CHECK: "organization": "Trail of Bits, inc.",
// CHECK: "product": "VAST",

int main(int argc, char **argv)
{
    return argc;
}
