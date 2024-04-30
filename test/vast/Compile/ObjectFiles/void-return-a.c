// RUN: %vast-front -c -vast-pipeline=with-abi -o %t.vast.o %s && %cc -c -xc %s.driver -o %t.clang.o  && %cc %t.vast.o %t.clang.o -o %t && (%t; test $? -eq 0)

void nothing(int a) {}

int vast_tests()
{
    // Simply test that call is correctly lowered.
    nothing( 5 );
    nothing( -15 );
    return 0;
}
