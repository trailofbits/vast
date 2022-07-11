// RUN: vast-cc --ccopts -xc --ccopts -std=c11 --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --ccopts -std=c11 --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.struct "v"
struct v {
   // CHECK: hl.union "v::anonymous[532]"
   union { // anonymous union
      // CHECK: hl.struct "v::anonymous[532]::anonymous[552]"
      // CHECK:    hl.field "i" : !hl.int
      // CHECK:    hl.field "j" : !hl.int
      // CHECK: hl.field "anonymous[609]" : !hl.named_type<<"v::anonymous[532]::anonymous[552]">>
      struct { int i, j; }; // anonymous structure
      // CHECK: hl.struct "v::anonymous[532]::anonymous[641]"
      // CHECK:   hl.field "k" : !hl.long
      // CHECK:   hl.field "l" : !hl.long
      // CHECK: hl.field "w" : !hl.named_type<<"v::anonymous[532]::anonymous[641]">>
      struct { long k, l; } w;
   };
   // CHECK: hl.field "anonymous[733]" : !hl.named_type<<"v::anonymous[532]">>
   // CHECK: hl.field "m" : !hl.int
   int m;
// CHECK: hl.var "v1" : !hl.lvalue<!hl.named_type<<"v">>>
} v1;

int main() {
   // CHECK: [[V1:%[0-9]+]] = hl.global.ref "v1" : !hl.lvalue<!hl.named_type<<"v">>>
   // CHECK: [[V2:%[0-9]+]] = hl.decl.ref [[V1]] : !hl.lvalue<!hl.named_type<<"v">>>

   // CHECK: [[V3:%[0-9]+]] = hl.member [[V2]] at "anonymous[733]" : !hl.lvalue<!hl.named_type<<"v">>> -> !hl.lvalue<!hl.named_type<<"v::anonymous[532]">>>
   // CHECK: [[V4:%[0-9]+]] = hl.member [[V3]] at "anonymous[609]" : !hl.lvalue<!hl.named_type<<"v::anonymous[532]">>> -> !hl.lvalue<!hl.named_type<<"v::anonymous[532]::anonymous[552]">>>
   // CHECK: [[V5:%[0-9]+]] = hl.member [[V4]] at "i" : !hl.lvalue<!hl.named_type<<"v::anonymous[532]::anonymous[552]">>> -> !hl.lvalue<!hl.int>
   // CHECK: [[C:%[0-9]+]] = hl.constant.int 2 : !hl.int
   // CHECK: hl.assign [[C]] to [[V5]] : !hl.int
   v1.i = 2;

   // CHECK: [[V1:%[0-9]+]] = hl.global.ref "v1" : !hl.lvalue<!hl.named_type<<"v">>>
   // CHECK: [[V2:%[0-9]+]] = hl.decl.ref [[V1]] : !hl.lvalue<!hl.named_type<<"v">>>
   // CHECK: [[V3:%[0-9]+]] = hl.member [[V2]] at "anonymous[733]" : !hl.lvalue<!hl.named_type<<"v">>> -> !hl.lvalue<!hl.named_type<<"v::anonymous[532]">>>
   // CHECK: [[V4:%[0-9]+]] = hl.member [[V3]] at "w" : !hl.lvalue<!hl.named_type<<"v::anonymous[532]">>> -> !hl.lvalue<!hl.named_type<<"v::anonymous[532]::anonymous[641]">>>
   // CHECK: [[V5:%[0-9]+]] = hl.member [[V4]] at "k" : !hl.lvalue<!hl.named_type<<"v::anonymous[532]::anonymous[641]">>> -> !hl.lvalue<!hl.long>
   v1.w.k = 5;
}
