## Lifting parameter passing [https://github.com/hsutter/708/blob/main/708.pdf]

The proposed way of parameter passing for next-gen c++ is to use declarative style:

```
f(     in X x) // x can be read from
f(  inout X x) // x can be used in read and write
f(    out X x) // x can be writen to
f(   move X x) // x will be moved from
f(forward X x) // x will be passed along
```

Similar holds for return values:

```
auto f()    move X { /* ... */ } // move X to caller
auto f()         X { /* ... */ } // possibly same
auto f() forward X { /* ... */ } // pass along X to the caller
```

Using the semantics aware vast dialects, we can design a method to automatically modernize code to use a declarative style of parameter passing.

## Examples

<table>
<tr>
<th>
CPP
</th>
<th>
CPP2
</th>
</tr>
<tr>
<td  valign="top">

<pre lang="cpp">
void f(const X& x) {
    g(x);
}
</pre>
</td>
<td  valign="top">

<pre lang="cpp">
void f(in X x) {
    g(x);
}
</pre>
</td>
</tr>
<tr>
<th>
VAST high-level dialect
</th>
<th>
Transformed to parameter dialect
</th>
</tr>
<tr>
<td  valign="top">

<pre lang="cpp">
hl.func @f(%x: !hl.ref< !hl.lvalue< !hl.struct< "X" > >, const >) {
    %0 = hl.call @g(%x) : (!hl.ref< !hl.lvalue< !hl.struct< "X" > >, const >) -> !hl.void
}
</pre>
</td>
<td  valign="top">

<pre lang="cpp">
hl.func @f(%x: !par.in< !hl.lvalue< !hl.struct< "X" > > >) {
    %0 = hl.call @g(%x) : (!par.in< !hl.lvalue< !hl.struct< "X" > > >) -> !hl.void
}
</pre>
</td>
</tr>
</table>

The transformation will be probably overapproximating, in cases when the analysis cannot determine the precise category, i.e., `inout` oveapproximates `in` and `out` parameters.

## Dialect

The dialect will define type adaptors for each parameter category:

```
!par.in< T >
!par.out< T >
!par.inout< T >
!par.move< T >
!par.forward< T >
```
Parameter categories can also be present as type attributes not to mess up the rest of the type trait system.
This needs further investigation.

The advantage of the type adaptors we can enforce the correct usage of values. For example, we can forbid usage of `out` parameter in other places than assignment.
