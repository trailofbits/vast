#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Decl.h>
#include <clang/AST/CanonicalType.h>
#include <llvm/ADT/FoldingSet.h>
#include <llvm/Support/TrailingObjects.h>
#include <mlir/IR/Types.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CallingConv.hpp"
#include "vast/CodeGen/Types.hpp"

namespace vast::cg
{
    /// Type for representing both the decl and type of parameters to a function.
    /// The decl must be either a ParmVarDecl or ImplicitParamDecl.
    struct function_arg_list : public llvm::SmallVector<const clang::VarDecl *, 16> {};

    /// abi_arg_info - Helper class to encapsulate information about how a specific C
    /// type should be passed to or returned from a function.
    struct abi_arg_info {
        enum class abi_arg_kind : std::uint8_t
        {
            /// Direct - Pass the argument directly using the normal converted vast type,
            /// or by coercing to another specified type stored in 'CoerceToType'). If
            /// an offset is specified (in UIntData), then the argument passed is offset
            /// by some number of bytes in the memory representation. A dummy argument
            /// is emitted before the real argument if the specified type stored in
            /// "PaddingType" is not zero.
            direct,

            /// Extend - Valid only for integer argument types. Same as 'direct' but
            /// also emit a zer/sign extension attribute.
            extend,

            /// Indirect - Pass the argument indirectly via a hidden pointer with the
            /// specified alignment (0 indicates default alignment) and address space.
            indirect,

            /// IndirectAliased - Similar to Indirect, but the pointer may be to an
            /// object that is otherwise referenced. The object is known to not be
            /// modified through any other references for the duration of the call, and
            /// the callee must not itself modify the object. Because C allows parameter
            /// variables to be modified and guarantees that they have unique addresses,
            /// the callee must defensively copy the object into a local variable if it
            /// might be modified or its address might be compared. Since those are
            /// uncommon, in principle this convention allows programs to avoid copies
            /// in more situations. However, it may introduce *extra* copies if the
            /// callee fails to prove that a copy is unnecessary and the caller
            /// naturally produces an unaliased object for the argument.
            indirect_aliased,

            /// Ignore - Ignore the argument (treat as void). Useful for void and empty
            /// structs.
            ignore,

            /// Expand - Only valid for aggregate argument types. The structure should
            /// be expanded into consecutive arguments for its constituent fields.
            /// Currently expand is only allowed on structures whose fields are all
            /// scalar types or are themselves expandable types.
            expand,

            /// CoerceAndExpand - Only valid for aggregate argument types. The structure
            /// should be expanded into consecutive arguments corresponding to the
            /// non-array elements of the type stored in CoerceToType.
            /// Array elements in the type are assumed to be padding and skipped.
            coerce_and_expand,

            // TODO: translate this idea to vast! Define it for now just to ensure that
            // we can assert it not being used
            in_alloca,
            kind_first = direct,
            kind_last  = in_alloca
        };

      private:

        struct direct_attr_info {
            unsigned offset;
            unsigned align;
        };

        struct indirect_attr_info {
            unsigned align;
            unsigned addr_space;
        };

        union {
            direct_attr_info direct_attr;     // is_direct() || is_extend()
            indirect_attr_info indirect_attr; // is_indirect()
            unsigned alloca_field_index;      // is_in_alloca()
        };

        abi_arg_kind kind;
        bool can_be_flattened : 1; // is_direct()
        bool signext          : 1; // is_extend()

        bool can_have_padding_type() const {
            return is_direct() || is_extend() || is_indirect() || is_indirect_aliased() || is_expand();
        }

      public:
        abi_arg_info(abi_arg_kind kind = abi_arg_kind::direct)
            : direct_attr{ 0, 0 }
            , kind(kind)
            , can_be_flattened(false)
        {}

        static abi_arg_info get_direct(
            unsigned offset       = 0,
            bool can_be_flattened = true,
            unsigned align        = 0
        ) {
            // FIXME constructor
            auto info = abi_arg_info(abi_arg_kind::direct);
            info.set_direct_offset(offset);
            info.set_direct_align(align);
            info.set_can_be_flattened(can_be_flattened);
            return info;
        }

        static abi_arg_info get_sign_extend(
            qual_type qtype, mlir_type type = nullptr
        ) {
            // FIXME constructor
            VAST_CHECK(qtype->isIntegralOrEnumerationType(), "Unexpected QualType");
            auto info = abi_arg_info(abi_arg_kind::extend);
            info.set_direct_offset(0);
            info.set_direct_align(0);
            info.set_sign_ext(true);
            return info;
        }

        static abi_arg_info get_zero_extend(
            qual_type qtype, mlir_type type = nullptr
        ) {
            // FIXME constructor
            VAST_CHECK(qtype->isIntegralOrEnumerationType(), "Unexpected QualType");
            auto info = abi_arg_info(abi_arg_kind::extend);
            info.set_direct_offset(0);
            info.set_direct_align(0);
            info.set_sign_ext(false);
            return info;
        }

        // abi_arg_info will record the argument as being extended based on the sign of
        // it's type.
        static abi_arg_info get_extend(qual_type qtype, mlir_type type = nullptr) {
            VAST_CHECK(qtype->isIntegralOrEnumerationType(), "Unexpected QualType");
            if (qtype->hasSignedIntegerRepresentation()) {
                return get_sign_extend(qtype, type);
            }
            return get_zero_extend(qtype, type);
        }

        static abi_arg_info get_ignore() { return abi_arg_info(abi_arg_kind::ignore); }

        abi_arg_kind get_kind() const { return kind; }
        bool is_direct() const { return kind == abi_arg_kind::direct; }
        bool is_in_alloca() const { return kind == abi_arg_kind::in_alloca; }
        bool is_extend() const { return kind == abi_arg_kind::extend; }
        bool is_indirect() const { return kind == abi_arg_kind::indirect; }
        bool is_indirect_aliased() const { return kind == abi_arg_kind::indirect_aliased; }
        bool is_expand() const { return kind == abi_arg_kind::expand; }
        bool is_coerce_and_expand() const { return kind == abi_arg_kind::coerce_and_expand; }

        bool can_have_coerce_to_type() const {
            return is_direct() || is_extend() || is_coerce_and_expand();
        }

        // Direct/Extend accessors
        unsigned get_direct_offset() const {
            VAST_CHECK((is_direct() || is_extend()), "Not a direct or extend kind");
            return direct_attr.offset;
        }

        void set_direct_offset(unsigned offset) {
            VAST_CHECK((is_direct() || is_extend()), "Not a direct or extend kind");
            direct_attr.offset = offset;
        }

        void set_direct_align(unsigned align) {
            VAST_CHECK((is_direct() || is_extend()), "Not a direct or extend kind");
            direct_attr.align = align;
        }

        void set_sign_ext(bool sext) {
            VAST_CHECK(is_extend(), "Invalid kind!");
            signext = sext;
        }

        void set_can_be_flattened(bool flatten) {
            VAST_CHECK(is_direct(), "Invalid kind!");
            can_be_flattened = flatten;
        }

        bool get_can_be_flattened() const {
            VAST_CHECK(is_direct(), "Invalid kind!");
            return can_be_flattened;
        }
    };


    struct function_info_arg_info {
        qual_type type;
        abi_arg_info info;
    };

    struct require_all_args_t {};

    constexpr auto require_all_args = require_all_args_t {};

    // A class for recording the number of arguments that a function signature
    // requires.
    class required_args {
        // The number of required arguments, or ~0 if the signature does not permit
        // optional arguments.
        unsigned num_required;

      public:

        required_args(require_all_args_t) : num_required(~0U) {}
        explicit required_args(unsigned n) : num_required(n) { VAST_ASSERT(n != ~0U); }

        unsigned get_opaque_data() const { return num_required; }

        bool allows_optional_args() const { return num_required != ~0U; }

        // Compute the arguments required by the given formal prototype, given that
        // there may be some additional, non-formal arguments in play.
        //
        // If function decl is not null, this will consider pass_object_size
        // params in function decl.
        static required_args for_prototype_plus(
            const clang::FunctionProtoType *prototype, unsigned /* additional */
        ) {
            VAST_UNIMPLEMENTED_IF(prototype->isVariadic());
            return { require_all_args };
        }

        static required_args for_prototype_plus(
            clang::CanQual< clang::FunctionProtoType > prototype, unsigned additional
        ) {
            return for_prototype_plus(prototype.getTypePtr(), additional);
        }

        unsigned get_num_required_args() const {
            VAST_ASSERT(allows_optional_args());
            return num_required;
        }
    };

    template< typename info_t >
    using info_trailing_object = llvm::TrailingObjects<
        info_t, function_info_arg_info, ext_param_info
    >;

    using llvm_folding_set_node = llvm::FoldingSetNode;

    struct function_info_t final
        : llvm_folding_set_node, info_trailing_object< function_info_t >
    {
      private:
        using arg_info = function_info_arg_info;

        friend info_trailing_object< function_info_t >;

        /// The vast::calling_conv to use for this function (as specified by the user).
        calling_conv calling_convention;

        /// The vast::calling_conv to actually use for this function, which may depend
        /// on the ABI.
        calling_conv effective_calling_convention : 8;

        /// The clang::calling_conv that this was originally created with.
        unsigned ast_calling_convention : 6;

        /// Whether this is an instance method.
        unsigned instance_method : 1;

        /// Whether this is a chain call.
        unsigned chain_call : 1;

        /// Whether this function is a CMSE nonsecure call
        unsigned cmse_nonsecure_call : 1;

        /// Whether this function is noreturn.
        unsigned no_return : 1;

        /// Whether this function is returns-retained.
        unsigned returns_retained : 1;

        /// Whether this function saved caller registers.
        unsigned no_caller_saved_regs : 1;

        /// How many arguments to pass inreg.
        unsigned has_reg_parm : 1;
        unsigned reg_parm    : 3;

        /// Whether this function has nocf_check attribute.
        unsigned no_cf_check : 1;

        required_args required;

        unsigned arg_struct_align       : 31;
        unsigned has_ext_parameter_infos : 1;

        unsigned num_args;

        arg_info *get_args_buffer() { return this->getTrailingObjects< arg_info >(); }
        const arg_info *get_args_buffer() const { return this->getTrailingObjects< arg_info >(); }

        ext_param_info *get_ext_param_infos_buffer() {
            return this->getTrailingObjects< ext_param_info >();
        }

        const ext_param_info *get_ext_param_infos_buffer() const {
            return this->getTrailingObjects< ext_param_info >();
        }

        explicit function_info_t() : required(require_all_args) {}

      public:
        static function_info_t *create(
            calling_conv calling_convention,
            bool instance_method,
            bool chain_call,
            const ext_info &ext_info,
            ext_parameter_info_span params,
            qual_type rty,
            qual_types_span arg_types,
            required_args required
        );

        void operator delete(void *p) { ::operator delete(p); }

        size_t numTrailingObjects(OverloadToken< arg_info >) const { return num_args + 1; }
        size_t numTrailingObjects(OverloadToken< ext_param_info >) const {
            return (has_ext_parameter_infos ? num_args : 0);
        }

        using const_arg_iterator = const arg_info *;
        using arg_iterator       = arg_info *;

        static void Profile(
            llvm::FoldingSetNodeID &id,
            bool instance_method,
            bool /* chain_call */,
            const ext_info &info,
            ext_parameter_info_span params,
            required_args required,
            qual_type rty,
            qual_types_span arg_types
        ) {
            id.AddInteger(info.getCC());
            id.AddBoolean(instance_method);
            // maybe fix? id.AddBoolean(chain_call);
            id.AddBoolean(info.getNoReturn());
            id.AddBoolean(info.getProducesResult());
            id.AddBoolean(info.getNoCallerSavedRegs());
            id.AddBoolean(info.getHasRegParm());
            id.AddBoolean(info.getRegParm());
            id.AddBoolean(info.getNoCfCheck());
            id.AddBoolean(info.getCmseNSCall());
            id.AddBoolean(required.get_opaque_data());
            id.AddBoolean(!params.empty());

            if (!params.empty()) {
                for (auto param : params) {
                    id.AddInteger(param.getOpaqueValue());
                }
            }

            rty.Profile(id);
            for (auto i : arg_types) {
                i.Profile(id);
            }
        }

        clang::CallingConv get_ast_calling_convention() const {
            return clang::CallingConv(ast_calling_convention);
        }

        void Profile(llvm::FoldingSetNodeID &id) {
            id.AddInteger(get_ast_calling_convention());
            id.AddBoolean(instance_method);
            id.AddBoolean(chain_call);
            id.AddBoolean(no_return);
            id.AddBoolean(returns_retained);
            id.AddBoolean(no_caller_saved_regs);
            id.AddBoolean(has_reg_parm);
            id.AddBoolean(reg_parm);
            id.AddBoolean(no_cf_check);
            id.AddBoolean(cmse_nonsecure_call);
            id.AddInteger(required.get_opaque_data());
            id.AddBoolean(has_ext_parameter_infos);

            if (has_ext_parameter_infos) {
                for (auto param : get_ext_param_infos()) {
                    id.AddInteger(param.getOpaqueValue());
                }
            }

            get_return_type().Profile(id);
            for (const auto &i : arguments()) {
                i.type.Profile(id);
            }
        }

        llvm::MutableArrayRef< arg_info > arguments() {
            return llvm::MutableArrayRef< arg_info >(arg_begin(), num_args);
        }

        llvm::ArrayRef< arg_info > arguments() const {
            return llvm::ArrayRef< arg_info >(arg_begin(), num_args);
        }

        const_arg_iterator arg_begin() const { return get_args_buffer() + 1; }
        const_arg_iterator arg_end() const { return get_args_buffer() + 1 + num_args; }
        arg_iterator arg_begin() { return get_args_buffer() + 1; }
        arg_iterator arg_end() { return get_args_buffer() + 1 + num_args; }

        unsigned arg_size() const { return num_args; }

        ext_parameter_info_span get_ext_param_infos() const {
            if (!has_ext_parameter_infos)
                return {};
            return {get_ext_param_infos_buffer(), num_args};
        }

        ext_param_info get_ext_param_infos(unsigned arg_idx) const {
            VAST_ASSERT(arg_idx <= num_args);
            if (!has_ext_parameter_infos)
                return ext_param_info();
            return get_ext_param_infos()[arg_idx];
        }

        // get_calling_convention - Return the user specified calling convention, which
        // has been translated into a vast calling conv.
        calling_conv get_calling_convention() const { return calling_convention; }

        qual_type get_return_type() const { return get_args_buffer()[0].type; }

        abi_arg_info &get_return_info() { return get_args_buffer()[0].info; }
        const abi_arg_info &get_return_info() const { return get_args_buffer()[0].info; }

        bool is_chain_call() const { return chain_call; }

        bool is_variadic() const { return required.allows_optional_args(); }
        required_args get_required_args() const { return required; }
        unsigned get_num_required_args() const {
            VAST_UNIMPLEMENTED_IF(is_variadic());
            return is_variadic() ? get_required_args().get_num_required_args() : arg_size();
        }

        // mlir::vast::StructType *getArgStruct() const { return ArgStruct; }

        /// Return true if this function uses inalloca arguments.
        // bool uses_in_alloca() const { return arg_struct; }
    };

} // namespace vast::cg
