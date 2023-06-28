// RUN: vast-front -o %t %s && (%t; test $? -eq 121)
// RUN: vast-front -o %t %s && (%t hello; test $? -eq 108)

int third( const char *arr )
{
    return arr[ 2 ];
}

int main(int argc, char **argv)
{
    if ( argc <= 1 )
        return 121;
    return third( argv[ 1 ] );
}
