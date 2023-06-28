// RUN: vast-front -o %t %s && (%t; test $? -eq 8)
int third( const char *arr )
{
    return arr[ 2 ];
}

int main()
{
    // 'l' - 'd' = 108 - 100
    return third("Hello")- third("ddd");
}
