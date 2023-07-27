// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.class "A" :
class A {
// CHECK: hl.access protected
protected:
    // CHECK: hl.dtor external virtual @_ZN1AD1Ev ()
    virtual ~A();
};

// CHECK: hl.class "B" :
class B {};

// CHECK: hl.class "C" :
class C : public A, protected virtual B {
    // CHECK: hl.base !hl.elaborated<!hl.record<"A">> public
    // CHECK: hl.base !hl.elaborated<!hl.record<"B">> protected virtual

// CHECK: hl.access public
public:
    // CHECK: hl.field "x" : !hl.int
    int x;

    // CHECK: %0 = hl.var "C::y" sc_static : !hl.lvalue<!hl.int>
    static int y;

    // CHECK: hl.method external virtual ref_none @_ZN1C5get_xEv () -> !hl.int
    virtual int get_x();

    // CHECK: hl.method external ref_none const @_ZNK1C5get_yEv () -> !hl.int
    int get_y() const;

    // CHECK: hl.method external ref_none volatile @_ZNV1C5get_zEv () -> !hl.int
    int get_z() volatile;

    // CHECK: hl.method external ref_lvalue @_ZNR1C5get_wEv () -> !hl.int
    int get_w() &;

    // CHECK: hl.method external ref_rvalue @_ZNO1C5get_pEv () -> !hl.int
    int get_p() &&;

    // CHECK: hl.ctor external @_ZN1CC1Ei (!hl.lvalue<!hl.int>)
    C(int x);
};