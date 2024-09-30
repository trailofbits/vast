/*
 * Copyright (c) 2021 Trail of Bits, Inc.
 */

#pragma once

#include <utility>
VAST_RELAX_WARNINGS
#include <mlir/IR/TypeSupport.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

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

        template< typename list, typename type >
        constexpr std::size_t index_of()
        {
            if constexpr ( list::empty ) {
                VAST_FATAL( "type not present in the list" );
            } else {
                using head = typename list::head;

                if (std::is_same_v< head, type >) {
                    return 0;
                }
                return 1 + index_of< typename list::tail, type >();
            }
        }

        template< typename list >
        constexpr bool is_unique() {
            if constexpr ( list::empty ) {
                return true;
            } else {
                using head = typename list::head;
                return !list::tail::template contains< head > && is_unique< typename list::tail >();
            }
        }

        template< typename list >
        struct flatten;

        template< typename ...types >
        struct flatten< type_list< types... > >
        {
            using type = concat< types... >::type;
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

        static constexpr bool empty = size == 0;

        using front      = typename detail::front< self >::type;
        using pop_front  = typename detail::pop_front< self >::type;

        using head = typename self::front;
        using tail = typename self::pop_front;

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

        template< typename type >
        static constexpr bool contains = (std::is_same_v< types, type > || ...);

        template< typename type >
        static constexpr std::size_t index_of = detail::index_of< self, type >();
    };

    template< typename list >
    concept empty_list = list::empty;

    template< typename type >
    struct is_type_list : std::false_type {};

    template< typename ...types >
    struct is_type_list< type_list< types... > > : std::true_type {};

    template< typename list >
    constexpr bool is_type_list_v = is_type_list< list >::value;

    template< typename list >
    constexpr bool is_list_of_lists = is_type_list_v< list > && !list::empty && list::template all_of< is_type_list >;

    template< typename list >
    concept list_of_lists = is_list_of_lists< list >;

    template< typename list >
    concept flat_list = !is_list_of_lists< list >;

    template< typename ...lists >
    using concat = typename detail::concat< lists... >::type;

    template< typename list >
    using materialized = detail::materialize< list >;

    template< typename ...types >
    using make_list = type_list< types... >;

    template< typename list >
    constexpr bool is_unique = detail::is_unique< list >();

    template< list_of_lists list >
    using flatten = detail::flatten< list >::type;

    namespace detail
    {

        template< typename derived >
        struct is_mlir_type
        {
            static constexpr bool value = std::is_base_of_v< mlir_type, derived >;
        };

        template< typename list, typename elem, std::size_t ...idxs >
        constexpr bool is_one_of(elem e, std::index_sequence< idxs... >)
        {
            return (isa< std::tuple_element_t< idxs, list > >(e) || ...);
        }

        template< typename list, typename elem >
        constexpr bool is_one_of(elem e)
        {
            // FIXME: use type_list::at
            return is_one_of< typename list::as_tuple >(
                e, std::make_index_sequence< list::size >{}
            );
        }

        template< typename list, typename ret, typename fn >
        constexpr ret dispatch(mlir_type type, fn &&f)
        {
            if constexpr ( list::empty ) {
                VAST_FATAL( "missing type to dispatch" );
            } else {
                using head = typename list::head;

                if (auto ty = mlir::dyn_cast< head >(type)) {
                    return f(ty);
                }

                return dispatch< typename list::tail, ret >(type, std::forward< fn >(f));
            }
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

    template< typename list, typename elem >
    constexpr bool is_one_of(elem e) { return detail::is_one_of< list >(e); }

    template< typename list, typename ret, typename fn >
    constexpr auto dispatch(mlir_type type, fn &&f)
    {
        return detail::dispatch< list, ret >(type, std::forward< fn >(f));
    }

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

        static_assert( type_list< int, double >::contains< int > );
        static_assert( !type_list< char, double >::contains< int > );

        static_assert( is_list_of_lists< type_list< type_list<> > > );
        static_assert( is_list_of_lists< type_list< type_list<>, type_list<> > > );
        static_assert( !is_list_of_lists< type_list< int > > );
        static_assert( !is_list_of_lists< type_list< int, type_list<> > > );
        static_assert( !is_list_of_lists< type_list< type_list<>, int > > );
        static_assert( !is_list_of_lists< type_list< int, int > > );
        static_assert( is_list_of_lists< type_list< type_list< int >, type_list< char, int > > > );

        using test_list = type_list< int, char, double >;
        static_assert( test_list::index_of< int > == 0 );
        static_assert( test_list::index_of< char > == 1 );
        static_assert( test_list::index_of< double > == 2 );

        static_assert( is_unique< type_list<> > );
        static_assert( is_unique< type_list< int > > );
        static_assert( is_unique< type_list< int, char > > );
        static_assert( !is_unique< type_list< int, char, int > > );
        static_assert( !is_unique< type_list< char, int, int > > );

        static_assert( std::is_same_v< flatten< type_list< type_list< int > > >, type_list< int > > );
    } // namespace test

} // namespace vast::util
