// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -
typedef struct TValue { struct TString *value_ ; unsigned char tt_ ; } TValue ;

typedef struct TString { char contents [ 1 ] ; } TString ;

static void kname ( const TValue * p , const char * * name ) {
    // CHECK: hl.cond {
    // CHECK: } ? {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.lvalue<!hl.ptr<!hl.char>>
    // CHECK: } : {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.ptr<!hl.char>
    // CHECK: } : !hl.ptr<!hl.char>
    * name = ( ( p -> tt_ ) & 0x0F ) == 4 ? ( p -> value_ ) -> contents : "?" ;
}
