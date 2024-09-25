// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @vp, <external> : !hl.lvalue<!hl.ptr<!hl.void>>
void * vp = 0;

// CHECK: hl.var @vpp, <external> : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.void>>>
void ** vpp = 0;

// CHECK: hl.var @ip, <external> : !hl.lvalue<!hl.ptr<!hl.int>>
int * ip = 0;

// CHECK: hl.var @icp, <external> : !hl.lvalue<!hl.ptr<!hl.int< const >>>
int const * icp = 0;

// CHECK: hl.var @cip, <external> : !hl.lvalue<!hl.ptr<!hl.int< const >>>
const int * cip = 0;

// CHECK: hl.var @ipc, <external> : !hl.lvalue<!hl.ptr<!hl.int,  const >>
int * const ipc = 0;

// CHECK: hl.var @icpc, <external> : !hl.lvalue<!hl.ptr<!hl.int< const >,  const >>
int const * const icpc = 0;

// CHECK: hl.var @cipc, <external> : !hl.lvalue<!hl.ptr<!hl.int< const >,  const >>
const int * const cipc = 0;

// CHECK: hl.var @ipp, <external> : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int>>>
int ** ipp = 0;

// CHECK: hl.var @ippc, <external> : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int>,  const >>
int ** const ippc = 0;

// CHECK: hl.var @ipcp, <external> : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int,  const >>>
int * const * ipcp = 0;

// CHECK: hl.var @icpp, <external> : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int< const >>>>
int const ** icpp = 0;

// CHECK: hl.var @ipcpc, <external> : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int,  const >,  const >>
int * const * const ipcpc = 0;

// CHECK: hl.var @ippp, <external> : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.ptr<!hl.int>>>>
int *** ippp = 0;

// CHECK: hl.var @uip, <external> : !hl.lvalue<!hl.ptr<!hl.int< unsigned >>>
unsigned int *uip = 0;
