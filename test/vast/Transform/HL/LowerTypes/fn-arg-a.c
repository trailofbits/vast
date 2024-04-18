// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types | %file-check %s
typedef enum { CURLE_OK = 0 } CURLcode ;

static CURLcode base64_encode(const char * table64, const char * inputbuff,
                              unsigned long insize, char * * outptr, unsigned long * outlen);

static const char base64encdec [] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";

// CHECK: hl.func @Curl_base64_encode {{.*}} (%arg0: !hl.lvalue<!hl.ptr<si8>>, %arg1: !hl.lvalue<ui64>, %arg2: !hl.lvalue<!hl.ptr<!hl.ptr<si8>>>, %arg3: !hl.lvalue<!hl.ptr<ui64>>) -> !hl.elaborated<!hl.typedef<"CURLcode">> {
CURLcode Curl_base64_encode (const char *inputbuff, unsigned long insize,
                             char **outptr, unsigned long *outlen)
{
    // CHECK: {{.*}} = hl.ref %arg0 : (!hl.lvalue<!hl.ptr<si8>>) -> !hl.lvalue<!hl.ptr<si8>>
    // CHECK: {{.*}} = hl.ref %arg1 : (!hl.lvalue<ui64>) -> !hl.lvalue<ui64>
    // CHECK: {{.*}} = hl.ref %arg2 : (!hl.lvalue<!hl.ptr<!hl.ptr<si8>>>) -> !hl.lvalue<!hl.ptr<!hl.ptr<si8>>>
    // CHECK: {{.*}} = hl.ref %arg3 : (!hl.lvalue<!hl.ptr<ui64>>) -> !hl.lvalue<!hl.ptr<ui64>>


    return base64_encode(base64encdec, inputbuff, insize, outptr, outlen) ;
}
