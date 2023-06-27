// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.class "A" :
class A {
public:
    virtual int get_x() = 0;
    virtual ~A() = default;
};

// CHECK: hl.class "B" :
class B {};

// CHECK: hl.class "C" :
class C : public A, protected virtual B {
    // CHECK: hl.base !hl.elaborated<!hl.record<"A">> public
    // CHECK: hl.base !hl.elaborated<!hl.record<"B">> protected virtual

    friend class D;
    friend int foobar();
    friend int A::get_x();

// CHECK: hl.access public
public:
    // CHECK: hl.field "x" : !hl.int
    int x;

    // CHECK: %0 = hl.var "C::y" sc_static : !hl.lvalue<!hl.int>
    static int y;

    int get_x() override;

    static int get_y() {
        return y;
    }

    int operator++(void) {
        return x;
    }

    int operator++(int) {
        return x;
    }

    C(int x);

    ~C();
};

C::C(int x) : x(x) {}

C::~C() {}

int C::get_x() {
    return x;
}