// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-to-hl-builtin | %file-check %s

#include <stdarg.h>

double average(int count, ...)
{
    va_list ap;
    int j;
    double tot = 0;
    // CHECK: hlbi.va_start
    va_start(ap, count); //Requires the last fixed parameter (to get the address)
    for(j=0; j<count; j++)
    // CHECK: hl.va_arg_expr
        tot+=va_arg(ap, double); //Requires the type to cast to. Increments ap to the next argument.
    // CHECK: hlbi.va_end
    va_end(ap);
    return tot/count;
}
