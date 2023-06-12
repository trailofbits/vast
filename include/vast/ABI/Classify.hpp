// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/Alignment.h>
VAST_UNRELAX_WARNINGS

#include "vast/ABI/ABI.hpp"

namespace vast::abi
{
    template< typename H, typename ... Tail >
    auto maybe_strip( mlir::Type t )
    {
        auto casted = [ & ]() -> mlir::Type
        {
            auto c = t.dyn_cast< H >();
            if ( c )
                return c.getElementType();
            return t;
        }();

        if constexpr ( sizeof ... ( Tail ) == 0 )
            return casted;
        else
            return maybe_strip< Tail ... >( casted );
    }

    struct TypeConfig
    {
        static bool is_void( mlir::Type t ) { return t.isa< mlir::NoneType >(); }

        static bool is_aggregate( mlir::Type t )
        {
            // TODO(abi): `SubElementTypeInterface` is not good enough, since for
            //             example hl::PointerType implements it.
            // TODO(abi): Figure how to better handle this than manual listing.
            return t.isa< hl::RecordType >()
                   || t.isa< hl::ElaboratedType >()
                   || t.isa< hl::ArrayType >();
        }

        static bool is_scalar( mlir::Type t )
        {
            // TODO(abi): Also complex is an option in ABI.
            return !is_aggregate( t );
        }

        static bool is_complex( mlir::Type t )
        {
            VAST_ASSERT( false );
        }

        static bool is_record( mlir::Type t )
        {
            if ( t.isa< hl::RecordType >() )
               return true;
            if ( auto et = t.dyn_cast< hl::ElaboratedType >() )
                return is_record( et.getElementType() );
            return false;
        }

        static bool is_struct( mlir::Type t )
        {
            // TODO(abi): Are these equivalent?
            return is_record( t );
        }

        static bool is_array( mlir::Type t )
        {
            return maybe_strip< hl::ElaboratedType >( t ).isa< hl::ArrayType >();
        }

        static bool can_be_passed_in_regs( mlir::Type t )
        {
            // TODO(abi): Seems like in C nothing can prevent this.
            return true;
        }

        static bool is_scalar_integer( mlir::Type t )
        {
            return t.isa< mlir::IntegerType >();
        }

        // TODO(abi): Implement.
        static bool has_unaligned_field( mlir::Type t )
        {
            return false;
        }

        static bool is_scalar_float( mlir::Type t )
        {
            return t.isa< mlir::FloatType >();
        }

        static bool is_pointer( mlir::Type t )
        {
            return t.isa< hl::PointerType >();
        }

        static hl::StructDeclOp get_struct_def( hl::RecordType t, mlir::ModuleOp m )
        {
            for ( auto &op : *m.getBody() )
                if ( auto decl = mlir::dyn_cast< hl::StructDeclOp >( op ) )
                    if ( decl.getName() == t.getName() )
                        return decl;
            return {};
        }

        static bool can_be_promoted( mlir::Type t )
        {
            return false;
        }

        static mlir::Type prepare( mlir::Type t )
        {
            return maybe_strip< hl::LValueType >( t );
        }

        // TODO(abi): Will need to live in a different interface.
        static std::size_t pointer_size() { return 64; }

        static std::size_t size( const auto &dl, mlir::Type t )
        {
            return dl.getTypeSizeInBits( t );
        }

        // [ start, end )
        static bool bits_contain_no_user_data( mlir::Type t, std::size_t start,
                                               std::size_t end, const auto &ctx )
        {
            const auto &[ dl, op ] = ctx;

            if ( size( dl, t ) <= start )
                return true;

            if ( is_array( t ) )
                VAST_TODO( "bits_contain_no_user_data, array: {0}", t );

            if ( is_record( t ) )
            {
                // TODO(abi): CXXRecordDecl.
                std::size_t current = 0;
                for ( auto field : fields( t, op ) )
                {
                    if ( current >= end )
                        break;
                    if ( !bits_contain_no_user_data( field, current, end - current, ctx ) )
                        return false;

                    current += size( dl, t );
                }
                return true;
            }

            return false;
        }

        static mlir::Type iN( const auto &has_context, std::size_t s )
        {
            return mlir::IntegerType::get( has_context.getContext(), s );
        }

        static mlir::Type int_type_at_offset( mlir::Type t, std::size_t offset,
                                              mlir::Type root, std::size_t root_offset,
                                              const auto &ctx )
        {
            const auto &[ dl, _ ] = ctx;
            //if ( offset != 0 )
            //    VAST_TODO( "int_type_at_offset called with {0}.", offset );

            auto is_int_type = [ & ]( std::size_t trg_size )
            {
                const auto &[ dl_, _ ] = ctx;
                return is_scalar_integer( t ) && size( dl_, t ) == trg_size;
            };


            if ( ( is_pointer( t ) && pointer_size() == 64 ) ||
                   is_int_type( 64 ) )
            {
                return t;
            }

            if ( is_int_type( 8 ) || is_int_type( 16 ) || is_int_type( 32 ) )
            {
                // TODO(abi): Here should be check if `BitsContainNoUserData` - however
                //            for now it should be safe to always pretend to it being `false`?
                if ( bits_contain_no_user_data( root, offset + size( dl, t ),
                                                root_offset + 64, ctx ) )
                {
                    return t;
                }
            }

            // We need to extract a field on current offset.
            // TODO(abi): This is done differently than clang, since they seem to be using
            //            underflow? on offset?
            if ( is_struct( t ) && ( size( dl, t )  > 64 ) )
            {
                auto [ field, field_start ] = field_containing_offset( ctx, t, offset );
                VAST_ASSERT( field );
                return int_type_at_offset( field, field_start, root, root_offset, ctx );
            }

            if ( is_array( t ) )
                VAST_TODO( "int_type_at_offset in {0} (is_array_was_true", t );

            auto type_size = size( dl, root );
            VAST_CHECK( type_size != 0, "Unexpected empty field? Type: {0}", t );

            auto final_size = std::min< std::size_t >( type_size - ( root_offset * 8 ), 64 );
            return mlir::IntegerType::get( t.getContext(), final_size );
        }

        static auto field_containing_offset( const auto &ctx, mlir::Type t, std::size_t offset )
            -> std::tuple< mlir::Type, std::size_t >
        {
            const auto &[ dl, op ] = ctx;

            auto curr = 0;
            for ( auto field : fields( t, op ) )
            {
                if ( curr + size( dl, field ) > offset )
                    return { field, curr };
                curr += size( dl, field );
            }
            VAST_UNREACHABLE( "Did not find field at offset {0} in {1}", offset,t );

        }

        template< typename Op >
        static std::vector< mlir::Type > fields( mlir::Type t, Op func )
        {
            std::vector< mlir::Type > out;

            auto striped = maybe_strip< hl::ElaboratedType >( t );
            VAST_ASSERT( striped );
            auto casted = striped.dyn_cast< hl::RecordType >();
            VAST_ASSERT( casted );

            auto def = get_struct_def( casted, func->template getParentOfType< mlir::ModuleOp >() );
            VAST_ASSERT( def );

            auto &region = def.getRegion();
            VAST_ASSERT( region.hasOneBlock() );

            for ( auto &op : *region.begin() )
            {
                auto field = mlir::dyn_cast< hl::FieldDeclOp >( op );
                VAST_ASSERT( field );
                out.push_back( field.getType() );
            }

            return out;
        }
    };


    // Stateful - one object should be use per one function classification.
    // TODO(abi): For now this serves as x86_64, later core parts will be extracted.
    // TODO(codestyle): See if `arg_` and `ret_` can be unified - in clang they are separately
    //                  but it may be just a bad codestyle.
    template< typename FnInfo, typename DL >
    struct classifier_base
    {
        using self_t = classifier_base< FnInfo, DL >;
        using func_info = FnInfo;
        using data_layout = DL;

        using type = typename func_info::type;
        using types = typename func_info::types;

        func_info info;
        const data_layout &dl;

        static constexpr std::size_t max_gpr = 6;
        static constexpr std::size_t max_sse = 8;

        std::size_t needed_int = 0;
        std::size_t needed_sse = 0;

        classifier_base( func_info info,
                         const data_layout &dl )
            : info( std::move( info ) ), dl( dl )
        {}

        auto size( mlir::Type t )
        {
            return dl.getTypeSizeInBits( t );
        }

        auto align( type )
        {
            // TODO(abi): Implement into data layout of vast stack.
            return 0;
        }

        // Enum for classification algorithm.
        enum class Class : uint32_t
        {
            Integer = 0,
            SSE,
            SSEUp,
            X87,
            X87Up,
            ComplexX87,
            Memory,
            NoClass
        };
        using classification_t = std::tuple< Class, Class >;

        static std::string to_string( Class c )
        {
            switch( c )
            {
                case Class::Integer: return "Integer";
                case Class::SSE: return "SSE";
                case Class::SSEUp: return "SSEUp";
                case Class::X87: return "X87";
                case Class::X87Up: return "X87Up";
                case Class::ComplexX87: return "ComplexX87";
                case Class::Memory: return "Memory";
                case Class::NoClass: return "NoClass";
            }
        }

        static std::string to_string( classification_t c )
        {
            auto [ lo, hi ] = c;
            return "[ " + to_string( lo ) + ", " + to_string( hi ) + " ]";
        }

        static Class join( Class a, Class b )
        {
            if ( a == b )
                return a;

            if ( a == Class::NoClass )
                return b;
            if ( b == Class::NoClass )
                return a;

            if ( a == Class::Memory || b == Class::Memory )
                return Class::Memory;

            if ( a == Class::Integer || b == Class::Integer )
                return Class::Integer;

            auto use_mem = [&]( auto x )
            {
                return x == Class::X87 || x == Class::X87Up || x == Class::ComplexX87;
            };

            if ( use_mem( a ) || use_mem( b ) )
                return Class::Memory;

            return Class::SSE;
        }

        static classification_t join( classification_t a, classification_t b )
        {
            auto [ a1, a2 ] = a;
            auto [ b1, b2 ] = b;
            return { join( a1, b1 ), join( a2, b2 ) };
        }

        classification_t get_class( mlir::Type t )
        {
            if ( TypeConfig::is_void( t ) )
                return { Class::NoClass, Class::NoClass };

            if ( TypeConfig::is_scalar_integer( t ) )
            {
                // _Bool, char, short, int, long, long long
                if ( size( t ) <= 64 )
                    return { Class::Integer, Class::NoClass };
                // __int128
                return { Class::Integer, Class::Integer };
            }

            if ( TypeConfig::is_scalar_float( t ) )
            {
                // float, double, _Decimal32, _Decimal64, __m64
                if ( size( t ) <= 64 )
                    return { Class::Integer, {} };
                // __float128, _Decimal128, __m128
                return { Class::SSE, { Class::SSEUp } };

                // TODO(abi): __m256
                // TODO(abi): long double
            }

            // TODO(abi): complex
            // TODO(abi): complex long double

            return get_aggregate_class( t );
        }

        // TODO(abi): Refactor.
        auto mk_ctx() const { return std::make_tuple( dl, info.raw_fn ); }

        classification_t get_aggregate_class( mlir::Type t )
        {
            if ( size( t ) > 4 * 64 || TypeConfig::has_unaligned_field( t ) )
                return { Class::Memory, {} };
            // TODO(abi): C++ perks.

            auto fields = TypeConfig::fields( t, info.raw_fn );
            classification_t result = { Class::NoClass, Class::NoClass };
            std::size_t offset = 0;
            for ( std::size_t i = 0; i < fields.size(); ++i )
            {
                auto field_class = [ & ]() -> classification_t
                {
                    auto [ lo, hi ] = classify( fields[ i ] );
                    if ( offset < 8 * 8 )
                        return { lo, hi };
                    VAST_ASSERT( hi == Class::NoClass );
                    return { Class::NoClass, lo };
                }();
                result = join( result, field_class );
                offset += size( fields[ i ] );
            }

            return post_merge( t, result );
        }


        classification_t post_merge( mlir::Type t, classification_t c )
        {
            auto [ lo, hi ] = c;
            if ( lo == Class::Memory || hi == Class::Memory )
                return { Class::Memory, Class::Memory };

            // Introduced in some revision.
            if ( hi == Class::X87Up && lo != Class::X87 )
            {
                VAST_ASSERT( false );
                lo = Class::Memory;
            }

            if ( size( t ) > 128 && ( lo != Class::SSE || hi != Class::SSEUp ) )
            {
                VAST_ASSERT( false );
                lo = Class::Memory;
            }

            if ( hi == Class::SSEUp && lo != Class::SSE )
                return { lo, Class::SSE };
            return { lo, hi };
        }

        auto classify( mlir::Type raw )
        {
            auto t = TypeConfig::prepare( raw );
            return get_class( t );
        }

        using half_class_result = std::variant< arg_info, type, std::monostate >;

        // Both parts are passed as argument.
        half_class_result return_lo( type t, classification_t c )
        {
            auto [ lo, hi ] = c;
            // Check are taken from clang, to not diverge from their implementation for now.
            switch ( lo )
            {
                case Class::NoClass:
                {
                    if ( hi == Class::NoClass )
                        return { arg_info::make< ignore >() };
                    // Missing lo part.
                    VAST_ASSERT( hi == Class::SSE || hi == Class::Integer
                                                  || hi == Class::X87Up );
                    [[ fallthrough ]];
                }

                case Class::SSEUp:
                case Class::X87Up:
                    VAST_ASSERT( false );

                case Class::Memory:
                    // TODO(abi): Inject type.
                    return { arg_info::make< indirect >( type{} ) };

                case Class::Integer:
                {
                    auto target_type = TypeConfig::int_type_at_offset( t, 0, t, 0, mk_ctx() );
                    // TODO(abi): get integer type for the slice.
                    if ( TypeConfig::is_scalar_integer( t ) &&
                         TypeConfig::can_be_promoted( t ) )
                    {
                        return { arg_info::make< extend >( target_type ) };
                    }
                    return { target_type };
                }
                default:
                    VAST_ASSERT( false );
            }
        }

        half_class_result return_hi( type t, classification_t c )
        {
            auto [ lo, hi ] = c;
            switch ( hi )
            {
                case Class::Memory:
                case Class::X87:
                    VAST_ASSERT( false );
                case Class::ComplexX87:
                case Class::NoClass:
                    return { std::monostate{} };
                case Class::Integer:
                {
                    auto target_type = TypeConfig::int_type_at_offset( t, 8, t, 8, mk_ctx() );
                    if ( lo == Class::NoClass )
                        return { arg_info::make< direct >( target_type ) };
                    return { target_type };

                }
                case Class::SSE:
                case Class::SSEUp:
                case Class::X87Up:
                    VAST_ASSERT( false );
            }
        }

        // TODO(abi): Implement. Requires information about alignment and size of alloca.
        types combine_half_types( type lo, type hi )
        {
            auto hi_start = llvm::alignTo( size( lo ) / 8, size( hi ) / 8 );
            VAST_CHECK( hi_start != 0 && hi_start / 8 <= 8,
                        "{0} {1} {2}",
                        lo, hi, hi_start );

            // `hi` needs to start at later offsite - we need to add explicit padding
            // to the `lo` type.
            auto adjusted_lo = [ & ]() -> type
            {
                if ( hi_start == 8 )
                    return lo;
                // TODO(abi): float, half -> promote to double
                VAST_CHECK( TypeConfig::is_scalar_integer( lo ) ||
                            TypeConfig::is_pointer( lo ),
                            "Cannot adjust half type for pair {0}. {1}", lo, hi );
                return TypeConfig::iN( lo, 64 );
            }();

            return { adjusted_lo, hi };
        }

        arg_info resolve_classification( type t, half_class_result low, half_class_result high )
        {
            // If either returned a result it should be used.
            // TODO(abi): Should `high` be allowed to return `arg_info`?
            if ( auto out = get_if< arg_info >( &low ) )
                return std::move( *out );
            if ( auto out = get_if< arg_info >( &high ) )
                return std::move( *out );

            if ( holds_alternative< std::monostate >( high ) )
            {
                auto coerced_type = get_if< type >( &low );
                VAST_ASSERT( coerced_type );
                // TODO(abi): Pass in `coerced_type`.
                return arg_info::make< direct >( *coerced_type );
            }

            // Both returned types, we need to comboine them.
            auto lo_type = get_if< type >( &low );
            auto hi_type = get_if< type >( &high );
            VAST_ASSERT( lo_type && hi_type );

            auto res_type = combine_half_types( *lo_type, *hi_type );
            return arg_info::make< direct >( res_type );

        }

        arg_info classify_return( type t )
        {
            if ( TypeConfig::is_void( t ) )
                return arg_info::make< ignore >();

            if ( auto record = TypeConfig::is_record( t ) )
                if ( !TypeConfig::can_be_passed_in_regs( t ) )
                    return arg_info::make< indirect >( type{} );

            // Algorithm based on AMD64-ABI
            auto c = classify( t );

            auto low = return_lo( t, c );
            auto high = return_hi( t, c );
            return resolve_classification( t, std::move( low ), std::move( high ) );
        }

        // Integer, SSE
        using reg_usage = std::tuple< std::size_t, std::size_t >;
        // TODO: This is pretty convoluted.
        using arg_class = std::tuple< half_class_result, reg_usage >;

        half_class_result arg_lo( type t, classification_t c )
        {
            auto [ lo, hi ] = c;

            switch( lo )
            {
                case Class::NoClass:
                {
                    if ( hi == Class::NoClass )
                        return { arg_info::make< ignore >() };
                    VAST_ASSERT( hi == Class::SSE || hi == Class::Integer
                                                  || hi == Class::X87Up );
                    return { std::monostate{} };
                }
                case Class::Memory:
                case Class::X87:
                case Class::ComplexX87:
                {
                    // TODO: Indirect + usage of int reg.
                    VAST_TODO( "arg_lo::Memory,X87,ComplexX87" );
                }

                case Class::SSEUp:
                case Class::X87Up:
                {
                    // This should actually not happen.
                    VAST_ASSERT( false );
                }

                case Class::Integer:
                {
                    ++needed_int;
                    auto target_type = TypeConfig::int_type_at_offset( t, 0, t, 0, mk_ctx() );
                    // TODO(abi): Or enum.
                    if ( hi == Class::NoClass && TypeConfig::is_scalar_integer( target_type ) )
                    {
                        // TODO(abi): If enum, treat as underlying type.
                        if ( TypeConfig::can_be_promoted( target_type ) )
                            return { arg_info::make< extend >( target_type ) };
                    }

                    return { target_type };
                }

                case Class::SSE:
                {
                    VAST_TODO( "arg_lo::SSE" );
                }
            }
        }


        half_class_result arg_hi( type t, classification_t c )
        {
            auto [ lo, hi ] = c;
            switch( hi )
            {
                case Class::Memory:
                case Class::X87:
                case Class::ComplexX87:
                    VAST_UNREACHABLE( "Invalid classification for arg_hi: {0}", to_string(hi) );

                case Class::NoClass:
                      return { std::monostate{} };

                case Class::Integer:
                {
                    ++needed_int;
                    auto target_type = TypeConfig::int_type_at_offset( t, 8, t, 8, mk_ctx() );
                    if ( lo == Class::NoClass )
                        return { arg_info::make< direct >( target_type ) };
                    return { target_type };
                }

                case Class::X87Up:
                case Class::SSE:
                {
                    VAST_TODO( "arg_hi::X87Up,SSE" );
                }
                case Class::SSEUp:
                {
                    VAST_TODO( "arg_hi::SSEUp" );
                }
            }
        }

        arg_info classify_arg( type t )
        {
            auto c = classify( t );
            auto low = arg_lo( t, c );
            auto high = arg_hi( t, c );
            return resolve_classification( t, std::move( low ), std::move( high ) );
        }

        self_t &compute_abi()
        {
            info.add_return( classify_return( TypeConfig::prepare( info.return_type() ) ) );
            for ( auto arg : info.fn_type().getInputs() )
                info.add_arg( classify_arg( TypeConfig::prepare( arg ) ) );
            return *this;
        }

        func_info take()
        {
            return std::move( info );
        }

    };

} // namespace vast::abi
