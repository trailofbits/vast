// RUN: vast-cc --from-source %s | FileCheck %s

int main()
{
    // CHECK: [[V1:%[0-9]+]] = hl.constant( 0 : i32 ): !hl.int
    // CHECK: hl.var( x, [[V1]] ): !hl.int
    int x = 0;
    // CHECK: [[V2:%[0-9]+]] = hl.declref( @x ): !hl.int
    // CHECK: %3 = hl.var( y, [[V2]] ): !hl.int
    int y = x;
}
