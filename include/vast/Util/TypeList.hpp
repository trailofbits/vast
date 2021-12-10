/*
 * Copyright (c) 2021 Trail of Bits, Inc.
 */

#pragma once

#include <utility>
VAST_RELAX_WARNINGS
#include <mlir/IR/TypeSupport.h>
VAST_UNRELAX_WARNINGS

#include <type_traits>
#include <tuple>
#include <optional>

namespace vast::util {

    template< typename ...types >
    struct type_list;

    namespace detail {

        using empty_list = type_list<>;

        template< typename list > struct front {};
        template< typename list > struct pop_front {};
        template< typename head, typename list > struct push_front {};
        template< typename fn, typename list > struct apply {};
        template< typename list > struct materialize {};
        template< typename left, typename right > struct concat_list {};
        template< typename ...lists > struct concat {};

        template<>
        struct front< empty_list > { using type = std::nullopt_t; };

        template< typename head, typename ...tail >
        struct front< type_list< head, tail... > > { using type = head; };

        template<>
        struct pop_front< empty_list > { using type = empty_list; };

        template< typename head, typename ...tail >
        struct pop_front< type_list< head, tail... > >
        {
            using type = type_list< tail... >;
        };

        template< typename head, typename ...tail >
        struct push_front< head, type_list< tail... > >
        {
            using type = type_list< head, tail... >;
        };

        template< typename fn, typename ...types >
        struct apply< fn, type_list< types... > >
        {
            using type = type_list< typename fn::template type< types >... >;
        };

        template< typename ...types >
        struct materialize< type_list< types... > > : types... {};

        template< typename ...left, typename ...right >
        struct concat_list< type_list< left... >, type_list< right... > >
        {
            using type = type_list< left..., right... >;
        };

        template< typename list >
        struct concat< list > { using type = list; };

        template< typename list, typename ...rest >
        struct concat< list, rest... >
        {
            using type = typename concat_list< list, typename concat< rest... >::type >::type;
        };

    } // namespace detail

    //
    // generic type_list
    //
    template< typename ...types >
    struct type_list
    {
        using self = type_list;
        static constexpr std::size_t size = sizeof...(types);

        using front      = typename detail::front< self >::type;
        using pop_front  = typename detail::pop_front< self >::type;

        template< typename type >
        using push_front = typename detail::push_front< type, self >::type;

        template< typename fn >
        using apply = typename detail::apply< fn, self >::type;

        using as_tuple = std::tuple< types... >;

        template< std::size_t idx >
        using at = typename std::tuple_element< idx, as_tuple >::type;

        template< template< typename > typename pred >
        static constexpr bool any_of = (pred<types>::value || ...);

        template< template< typename > typename pred >
        static constexpr bool all_of = (pred<types>::value && ...);

        template< template< typename > typename pred >
        static constexpr bool none_of = !any_of<pred>;
    };

    template< typename ...lists >
    using concat = typename detail::concat< lists... >::type;

    template< typename list >
    using materialized = detail::materialize< list >;

    template< typename ...types >
    using make_list = type_list< types... >;

    namespace detail
    {
        using mlir_type = mlir::Type;

        template< typename derived >
        struct is_mlir_type
        {
            static constexpr bool value = std::is_base_of_v< mlir_type, derived >;
        };

        template< typename list, std::size_t ...idxs >
        constexpr bool is_one_of(mlir_type type, std::index_sequence< idxs... >)
        {
            return (type.isa< std::tuple_element_t< idxs, list > >() || ...);
        }

        template< typename list >
        constexpr bool is_one_of(mlir_type type)
        {
            // FIXME: use type_list::at
            return is_one_of< typename list::as_tuple >(
                type, std::make_index_sequence< list::size >{}
            );
        }

    } // namespace detail

    //
    // mlir specific type list
    //
    template< typename ...types >
    struct mlir_type_list : type_list< types... >
    {
        using base = type_list< types... >;
        static_assert( base::template all_of< detail::is_mlir_type > );
    };

    template< typename list >
    constexpr bool is_one_of(auto type) { return detail::is_one_of< list >(type); }

    namespace test
    {
        static_assert( std::is_same_v< type_list< int, char* >::front, int > );
        static_assert( std::is_same_v< type_list< char* >::front, char* > );

        static_assert( std::is_same_v< type_list< int, char* >::pop_front, type_list< char* > > );
        static_assert( std::is_same_v< type_list< int >::pop_front, type_list<> >);

        static_assert( std::is_same_v< type_list< int >::as_tuple, std::tuple< int > >);
        static_assert( std::is_same_v< type_list<>::as_tuple, std::tuple<> >);

        static_assert( std::is_same_v< type_list<>::push_front< int >, type_list< int > > );
        static_assert( std::is_same_v< type_list< int, void >::push_front< char >, type_list< char, int, void > > );

        static_assert( std::is_same_v< concat< type_list<>, type_list<> >, type_list<> > );
        static_assert( std::is_same_v< concat< type_list< int >, type_list<> >, type_list< int > > );
        static_assert( std::is_same_v< concat< type_list< void >, type_list< void > >, type_list< void, void > > );

        static_assert( std::is_same_v< type_list< int, char >::at< 0 >, int > );
        static_assert( std::is_same_v< type_list< int, char >::at< 1 >, char > );

        static_assert( type_list<>::size == 0 );
        static_assert( type_list< int, char >::size == 2 );

        struct mutate { template< typename T > using type = const T; };
        static_assert( std::is_same_v< type_list< int >::apply< mutate >, type_list< const int > > );

        template< typename type >
        struct is_int { static constexpr bool value = std::is_same_v< type, int >; };

        static_assert( !type_list<>::any_of< is_int > );
        static_assert( type_list< void, int >::any_of< is_int > );
        static_assert( !type_list< void, char >::any_of< is_int > );

        static_assert( type_list<>::all_of< is_int > );
        static_assert( type_list< int, void, float >::all_of< std::is_fundamental > );

        static_assert( type_list<>::none_of< is_int > );
        static_assert( type_list< int, void, float >::none_of< std::is_compound > );
    } // namespace test

} // namespace vast::util
