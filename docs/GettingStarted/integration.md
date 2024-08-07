
## Integration to custom tools

VAST is designed with a focus on highly customizable code generation, allowing
developers to tailor the process to their specific needs. At the heart of this customization is the `CodeGenDriver`.

## Using the Predefined Driver or Customizing Components

The `CodeGenDriver` offers the flexibility to either use a predefined driver or
customize each of its components according to the requirements of your project.
This driver is responsible for generating a single MLIR module, typically
corresponding to a single translation unit, although it can also be utilized for
partial code generation.

### Components of the CodeGen Driver

1. __Metadata (locations) generator__: The `vast::cg::meta_generator` serves as a
configuration point for creating locations attached to MLIR operations. It
exposes the following API:

```cpp
    struct meta_generator {
        virtual ~meta_generator() = default;
        virtual loc_t location(const clang_decl *) const = 0;
        virtual loc_t location(const clang_stmt *) const = 0;
        virtual loc_t location(const clang_expr *) const = 0;
    };
```

In VAST, two implementations are provided:
- `default_meta_gen`: Translates Clang AST locations to MLIR locations.
- `id_meta_gen`: Assigns consecutive IDs as meta information, providing an
alternative approach.

2. __Symbol (names) generator__: The `vast::cg::symbol_generator` serves as a
configuration point for naming of program symbols. It exposes the following API:

```cpp
    struct symbol_generator
    {
        virtual ~symbol_generator() = default;

        virtual std::optional< symbol_name > symbol(clang_global decl) = 0;
        virtual std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) = 0;
    };
```

The symbol generator handles name mangling and the generation of all symbol names in MLIR.
VAST provides a default symbol generator, `default_symbol_generator`, which implements
symbol mangling in a manner similar to Clang.

3. __AST Visitor__: Holds the main logic for MLIR generation. Visitor bundles
together both location generator and meta generator, it implements the following
API:

```cpp
    struct visitor_base
    {
        virtual ~visitor_base() = default;

        virtual operation visit(const clang_decl *, scope_context &scope) = 0;
        virtual operation visit(const clang_stmt *, scope_context &scope) = 0;
        virtual mlir_type visit(const clang_type *, scope_context &scope) = 0;
        virtual mlir_type visit(clang_qual_type, scope_context &scope)    = 0;
        virtual std::optional< named_attr > visit(const clang_attr *, scope_context &scope) = 0;

        virtual operation visit_prototype(const clang_function *decl, scope_context &scope) = 0;

        virtual std::optional< loc_t > location(const clang_decl *) = 0;
        virtual std::optional< loc_t > location(const clang_stmt *) = 0;
        virtual std::optional< loc_t > location(const clang_expr *) = 0;

        virtual std::optional< symbol_name > symbol(clang_global) = 0;
        virtual std::optional< symbol_name > symbol(const clang_decl_ref_expr *) = 0;
    };
```

Visitors can operate independently, but leveraging multiple visitors in chain
often offers advantages of their composition. In VAST, we provide several
specialized visitors:

- `default_visitor`: This visitor handles the default translation to a high-level dialect.
- `unsup_visitor`: Generates operations marked as "unsupported," serving as a fallback for default visitation.
- `unreach_visitor` (unreachable visitor): Yields an error upon visitation and is typically used as the last visitor in the chain.

The visitor API follows a design where its functions return optional results or
empty MLIR entities if visitation, location, or symbol generation fails. This
design facilitates easy recognition of such cases and enables the passing of
visitation to another visitor.

For cases where only partial visitor implementation is desired, developers can
utilize a helper class called `fallthrough_visitor` as a base for their visitor.
By default, it treats all visitations as unsuccessful but allows for overrides.
This empowers developers to implement specialized visitors, such as a dedicated
type visitor, for specific use cases.

To chain visitors, VAST offers a class to represent a list of visitors called
`vast::cg::visitor_list`. This list satisfies the visitor API and forwards
visitation to the head of the list, enabling cascading through its visitors
starting from the head.

Visitor list nodes serve a dual purpose: they can either solely hold visitors
and call them, or they can perform preprocessing on visitation arguments or
postprocessing on results.

The base class for a visitor list node, which wraps a visitor class, is defined
as follows:

```cpp
template <typename visitor>
visitor_list_node_adaptor
```

This class simply forwards visitation to the internal visitor without performing
any additional actions. Notably, it doesn't even forward the visitation to the
next node on failure.

For cases where a visitor may fail, another class is available:

```cpp
template <typename visitor>
try_or_through_list_node
```

This class attempts to call its visitor, and if it fails, it returns the
application of `next->visit` with same arguments.

Additionally, a helper class for building partial visitor list nodes is
provided:

```cpp
template <typename visitor>
fallthrough_list_node
```

This class, by default, forwards everything to the next visitor but allows for
overriding specific visitor methods as needed.

To simplify the construction of a chain of visitors, these classes override the `operator|`, enabling chaining. Additionally, we provide convenient constructor methods for specifying how to build each node:

```cpp
template <typename visitor, typename ...args_t> as_node;
```

This method creates a visitor list node with the given visitor and passes
arguments to its constructor. Sometimes, visitors need to invoke the head of the
visitor list on subelements of the currently visited entity. To achieve this,
`as_node_with_list_ref` can be used, which passes the reference as the first
argument to the constructor of the visitor. Finally, `optional` allows for
configuring which visitors to plug into the pipeline at runtime.

### Example: Default Visitor List Construction

The construction of the default VAST visitor appears as follows:

```cpp
auto visitors = std::make_shared<visitor_list>()
    | as_node_with_list_ref<attr_visitor_proxy>()
    | as_node<type_caching_proxy>()
    | as_node_with_list_ref<default_visitor>(
        mctx, actx, *bld, std::move(mg), std::move(sg), strict_return, missing_return_policy
    )
    | optional(enable_unsupported, as_node_with_list_ref<unsup_visitor>(*mctx, *bld))
    | as_node<unreach_visitor>();
```

In this construction:

- The first visitor is a proxy responsible for post-processing visitor results and attaching any applicable attributes from clang AST.
- The `type_caching_proxy` is utilized to deduplicate generated types. It either returns a stored cached type or forwards the visitation. This proxy enables reusing its cache to generate the data layout at the end of module generation.
- Next, we have the `default_visitor`, followed by an optional `unsupported_visitor`, and lastly, the `unreach_visitor`, which yields an error if reached.

### Simple Driver Integration

```cpp
// Example usage of the driver class

// Create a driver instance
driver drv(actx, mctx, bld, visitors);

// Emit declarations
drv.emit(decls);

// Finalize code generation -- this emits e.g., data layout information
drv.finalize();

// Verify the generated module
if (!drv.verify()) {
    // Handle verification failure
}

// Retrieve the finalized module
owning_mlir_module_ref mod = drv.freeze();
```
