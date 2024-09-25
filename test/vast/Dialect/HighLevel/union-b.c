// RUN: %vast-cc1 -vast-emit-mlir=hl -std=c11 %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl -std=c11 %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.struct @v
struct v {
   // CHECK: hl.union @"[[N1:anonymous\[[0-9]+\]]]"
   union { // anonymous union
      // CHECK: hl.struct @"[[N2:anonymous\[[0-9]+\]]]"
      // CHECK:    hl.field @i : !hl.int
      // CHECK:    hl.field @j : !hl.int
      // CHECK: hl.field @"[[N3:anonymous\[[0-9]+\]]]" : !hl.record<@"[[N2]]">
      struct { int i, j; }; // anonymous structure
      // CHECK: hl.struct @"[[N4:anonymous\[[0-9]+\]]]"
      // CHECK:   hl.field @k : !hl.long
      // CHECK:   hl.field @l : !hl.long
      // CHECK: hl.field @w : !hl.elaborated<!hl.record<@"[[N4]]">>
      struct { long k, l; } w;
   };
   // CHECK: hl.field @"[[N5:anonymous\[[0-9]+\]]]" : !hl.record<@"[[N1]]">
   // CHECK: hl.field @m : !hl.int
   int m;
// CHECK: hl.var @v1, <common> : !hl.lvalue<!hl.elaborated<!hl.record<@v>>>
} v1;

int main() {
   // CHECK: [[V1:%[0-9]+]] = hl.ref @v1 : !hl.lvalue<!hl.elaborated<!hl.record<@v>>>

   // CHECK: [[V2:%[0-9]+]] = hl.member [[V1]] at @"[[N5]]" : !hl.lvalue<!hl.elaborated<!hl.record<@v>>> -> !hl.lvalue<!hl.record<@"[[N1]]">>
   // CHECK: [[V3:%[0-9]+]] = hl.member [[V2]] at @"[[N3]]" : !hl.lvalue<!hl.record<@"[[N1]]">> -> !hl.lvalue<!hl.record<@"[[N2]]">>
   // CHECK: [[V4:%[0-9]+]] = hl.member [[V3]] at @i : !hl.lvalue<!hl.record<@"[[N2]]">> -> !hl.lvalue<!hl.int>
   // CHECK: [[C:%[0-9]+]] = hl.const #core.integer<2> : !hl.int
   // CHECK: hl.assign [[C]] to [[V4]] : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
   v1.i = 2;

   // CHECK: [[V1:%[0-9]+]] = hl.ref @v1 : !hl.lvalue<!hl.elaborated<!hl.record<@v>>>
   // CHECK: [[V2:%[0-9]+]] = hl.member [[V1]] at @"[[N5]]" : !hl.lvalue<!hl.elaborated<!hl.record<@v>>> -> !hl.lvalue<!hl.record<@"[[N1]]">>
   // CHECK: [[V3:%[0-9]+]] = hl.member [[V2]] at @w : !hl.lvalue<!hl.record<@"[[N1]]">> -> !hl.lvalue<!hl.elaborated<!hl.record<@"[[N4]]">>>
   // CHECK: [[V4:%[0-9]+]] = hl.member [[V3]] at @k : !hl.lvalue<!hl.elaborated<!hl.record<@"[[N4]]">>> -> !hl.lvalue<!hl.long>
   v1.w.k = 5;
}
