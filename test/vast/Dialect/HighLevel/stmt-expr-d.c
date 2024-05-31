// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

struct large { int x, y[9]; };

int main() {
    // CHECK: hl.member {{.*}} at "x" : !hl.elaborated<!hl.record<"large">> -> !hl.int
    int fixed = ({ struct large temp3; temp3.x = 2; temp3; }).x
	  - ({ struct large temp4; temp4.x = 1; temp4; }).x;
}
