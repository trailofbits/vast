// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @vp : !hl.lvalue<!hl.ptr<!hl.void>>
void * vp = 0;

// CHECK: hl.var @vpp : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.void>>>
void ** vpp = 0;

// CHECK: hl.var @ip : !hl.lvalue<!hl.ptr<!hl.int>>
int * ip = 0;

// CHECK: hl.var @icp : !hl.lvalue<!hl.ptr<!hl.int< const >>>
int const * icp = 0;

// CHECK: hl.var @cip : !hl.lvalue<!hl.ptr<!hl.int< const >>>
const int * cip = 0;

// CHECK: hl.var @ipc : !hl.lvalue<!hl.ptr<!hl.int,  const >>
int * const ipc = 0;

// CHECK: hl.var @icpc : !hl.lvalue<!hl.ptr<!hl.int< const >,  const >>
int const * const icpc = 0;

// CHECK: hl.var @cipc : !hl.lvalue<!hl.ptr<!hl.int< const >,  const >>
const int * const cipc = 0;

// CHECK: hl.var @ipp : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int>>>
int ** ipp = 0;

// CHECK: hl.var @ippc : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int>,  const >>
int ** const ippc = 0;

// CHECK: hl.var @ipcp : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int,  const >>>
int * const * ipcp = 0;

// CHECK: hl.var @icpp : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int< const >>>>
int const ** icpp = 0;

// CHECK: hl.var @ipcpc : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int,  const >,  const >>
int * const * const ipcpc = 0;

// CHECK: hl.var @ippp : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.ptr<!hl.int>>>>
int *** ippp = 0;

// CHECK: hl.var @uip : !hl.lvalue<!hl.ptr<!hl.int< unsigned >>>
unsigned int *uip = 0;
