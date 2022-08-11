// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -
// REQUIRES: qualifiers

// CHECK: !hl.ptr<!hl.void>
void * vp = 0;

// CHECK: !hl.ptr<!hl.ptr<!hl.void>>
void ** vpp = 0;

// CHECK: !hl.ptr<!hl.int>
int * ip = 0;

// CHECK: !hl.ptr<!hl.int<const>>
int const * icp = 0;

// CHECK: !hl.ptr<!hl.int<const>>
const int * cip = 0;

// CHECK: !hl.ptr<!hl.int, const>
int * const ipc = 0;

// CHECK: !hl.ptr<!hl.int<const>, const>
int const * const icpc = 0;

// CHECK: !hl.ptr<!hl.int<const>, const>
const int * const cipc = 0;

// CHECK: !hl.ptr<!hl.ptr<!hl.int>>
int ** ipp = 0;

// CHECK: !hl.ptr<!hl.ptr<!hl.int>, const>
int ** const ippc = 0;

// CHECK: !hl.ptr<!hl.ptr<!hl.int, const>>
int * const * ipcp = 0;

// CHECK: !hl.ptr<!hl.ptr<!hl.int<const>>>
int const ** icpp = 0;

// CHECK: !hl.ptr<!hl.ptr<!hl.int, const>, const>
int * const * const ipcpc = 0;

// CHECK: !hl.ptr<!hl.ptr<!hl.ptr<!hl.int>>>
int *** ippp = 0;

// CHECK: !hl.ptr<!hl.int<unsigned>>
unsigned int *uip = 0;
