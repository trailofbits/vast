// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types | %file-check %s -check-prefix=LTYPES
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-hl-lower-elaborated-types --vast-hl-lower-typedefs | %file-check %s -check-prefix=LTYPEDEFS

typedef void * ( * curl_malloc_callback ) ( unsigned long size ) ;

extern curl_malloc_callback Curl_cmalloc ;

void * malloc ( unsigned long __size );

curl_malloc_callback Curl_cmalloc = ( curl_malloc_callback ) malloc ;

// LTYPES: hl.typedef @curl_malloc_callback : !hl.ptr<!core.fn<(!hl.lvalue<ui64>) -> (!hl.ptr<ui8>)>>
// LTYPES: hl.var @Curl_cmalloc : !hl.lvalue<!hl.elaborated<!hl.typedef<"curl_malloc_callback">>> = {

// LTYPEDEFS: hl.var @Curl_cmalloc sc_extern : !hl.lvalue<!hl.ptr<!core.fn<(!hl.lvalue<ui64>) -> (!hl.ptr<ui8>)>>>
// LTYPEDEFS: hl.var @Curl_cmalloc : !hl.lvalue<!hl.ptr<!core.fn<(!hl.lvalue<ui64>) -> (!hl.ptr<ui8>)>>> = {
