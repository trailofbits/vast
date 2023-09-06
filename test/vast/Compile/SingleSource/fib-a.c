// RUN: %vast-front -o %t %s && (%t; test $? -eq 42)
int fib( int x )
{
    if ( x == 0 )
        return 0;
    if ( x <= 2 )
        return 1;

    return fib( x - 1 ) + fib( x - 2 );
}

int main()
{
    if ( fib( 0 ) != 0 )
        return 1;
    if ( fib( 3 ) != 2 )
        return 2;
    if ( fib( 4 ) != 3 )
        return 3;
    if ( fib( 5 ) != 5 )
        return 4;
    return 42;
}
