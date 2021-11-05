// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include <string_view>
#include <type_traits>
#include <optional>
#include <variant>
#include <concepts>

namespace vast
{
    template< typename F, typename... Args >
    concept invocable = requires(F &&f, Args&& ...args) {
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    };

    using parse_input_t = std::string_view;

    template< typename P >
    concept Parser = invocable<P, parse_input_t >;

    template< typename T >
    using parse_result_t = std::optional< std::pair< T, std::string_view > >;

    template< Parser P >
    using parse_invoke_result = std::invoke_result_t< P, parse_input_t >;

    template< Parser P >
    using parse_result_type = typename parse_invoke_result< P >::value_type;

    template< Parser P >
    using parse_type = typename parse_result_type< P >::first_type;

    template< typename P, typename T >
    concept parser = Parser< P > &&
        std::is_same_v< parse_invoke_result< P >, parse_result_t< T > >;

    template< typename T >
    using parser_t = auto (*)( parse_input_t ) -> parse_result_t< T >;

    constexpr const auto& result(const auto &p) { return p->first; }
    constexpr auto& result(auto &p) { return p->first; }
    constexpr auto result(auto &&p) { return std::move(p->first); }

    // --- monad parser functions ---

    // fmap applies function 'f' to parser result
    template< typename F, typename P, typename T = std::invoke_result_t< F, parse_type<P> > >
    constexpr parser<T> auto fmap(F &&f, P &&p)
    {
        return [f = std::forward< F >(f), p = std::forward< P >(p)] (parse_input_t in)
            -> parse_result_t< T >
        {
            if (auto res = p(in)) {
                const auto [val, rest] = res.value();
                return {{f(val), rest}};
            }
            return std::nullopt;
        };
    }

    template< typename P, typename F,
                typename T =  std::invoke_result_t< F, parse_type<P>, parse_input_t > >
    constexpr parser<T> auto bind(P&& p, F&& f)
    {
        return [=] (parse_input_t in) -> T {
            if (auto res = p(in)) {
                const auto [val, rest] = res.value();
                return f(val, rest);
            }
            return std::nullopt;
        };
    }

    // lifts a value into parser
    template< typename T >
    constexpr parser<T> auto lift(T &&t)
    {
        return [t = std::forward< T >(t)] (parse_input_t in) -> parse_result_t< T > {
            return {{t, in}};
        };
    }

    // always failing parser
    template< typename T >
    constexpr parser<T> auto fail(T)
    {
        return [=] (parse_input_t) -> parse_result_t< T > {
            return std::nullopt;
        };
    }

    template< typename T, typename Err >
    constexpr parser<T> auto fail(T&&, Err &&err)
    {
        return [=] (parse_input_t) -> parse_result_t< T > {
            err();
            return std::nullopt;
        };
    }

    // --- parser combinators ---

    // alternation: frirst try P1, and if it fails, try P2.
    // Both parsers have to return the same type.
    template< typename P1, typename P2,
                typename T = std::enable_if_t< std::is_same_v< parse_type<P1>, parse_type<P2> > > >
    constexpr parser<T> auto operator|(P1&& p1, P2&& p2)
    {
        return [=] (parse_input_t in) {
            if (auto r1 = p1(in))
                return r1;
            return p2(in);
        };
    }

    // accumulation: combine results of sequential application of both parsers
    template< typename P1, typename P2, typename F,
                typename T = std::invoke_result_t< F, parse_type<P1>, parse_type<P2> > >
    constexpr parser<T> auto combine(P1 &&p1, P2 &&p2, F&& f)
    {
        return [=] (parse_input_t in) -> parse_result_t< T > {
            if (auto r1 = p1(in))
                if (auto r2 = p2(rest(r1)))
                return {{f(result(r1), result(r2)), rest(r2)}};
            return std::nullopt;
        };
    }

    // accumulate: parser results to tuple
    template< typename P, typename T = std::tuple< parse_type<P> > >
    constexpr parser<T> auto combine(P &&p)
    {
        return [=] (parse_input_t in) -> parse_result_t< T > {
            if (auto r = p(in))
                return {{std::make_tuple(result(r)), rest(r)}};
            return std::nullopt;
        };
    }

    template< typename P, typename ...Ps
            , typename T = std::tuple< parse_type<P>, parse_type<Ps>... > >
    constexpr parser<T> auto combine(P &&p, Ps&&... ps)
    {
        return [=] (parse_input_t in) -> parse_result_t< T > {
            if (auto r1 = combine(p)(in))
                if (auto r2 = combine(ps...)(rest(r1)))
                return {{std::tuple_cat(result(r1), result(r2)), rest(r2)}};
            return std::nullopt;
        };
    }

    template< typename P1, typename P2, typename T = std::tuple< parse_type<P1>, parse_type<P2> > >
    constexpr parser<T> auto operator&(P1 &&p1, P2 &&p2)
    {
        return combine(std::forward<P1>(p1), std::forward<P2>(p2));
    }

    // combine two parsers and return the result of the second one
    template< typename P1, typename P2,
                typename L = parse_type<P1>, typename R = parse_type<P2> >
    constexpr parser<R> auto operator<(P1 &&p1, P2 &&p2)
    {
        return combine(std::forward<P1>(p1), std::forward<P2>(p2),
            [] (auto, const auto& r) { return r; });
    }

    // combine two parsers and return the result of the first one
    template< typename P1, typename P2,
                typename L = parse_type<P1>, typename R = parse_type<P2> >
    constexpr parser<L> auto operator>(P1 &&p1, P2 &&p2)
    {
        return combine(std::forward<P1>(p1), std::forward<P2>(p2),
            [] (const auto& l, auto) { return l; });
    }

    // try to apply a parser, and if it fails, return a default value
    template< typename P, typename T = parse_type<P> >
    constexpr parser<T> auto option(T &&def, P &&p)
    {
        return [p = std::forward<P>(p), def = std::forward<T>(def)] (parse_input_t in)
            -> parse_result_t< T >
        {
            if (auto res = p(in))
                return res;
            return {{def, in}};
        };
    }

    // parse character 'c'
    constexpr parser<char> auto char_parser(char c)
    {
        return [=] (parse_input_t in) -> parse_result_t<char> {
            if (in.empty() || in.front() != c)
                return std::nullopt;
            return {{c, in.substr(1)}};
        };
    }

    // parse string 'pattern'
    constexpr parser<std::string_view> auto string_parser(std::string_view pattern)
    {
        return [=] (parse_input_t in) -> parse_result_t<std::string_view> {
            if (in.starts_with(pattern))
                return {{pattern, in.substr(pattern.size())}};
            return std::nullopt;
        };
    }

    // parse character ∈ chars
    constexpr parser<char> auto one_of(std::string_view chars)
    {
        return [=] (parse_input_t in) -> parse_result_t<char> {
            if (in.empty())
                return std::nullopt;
            if (chars.find(in.front()) != chars.npos)
                return {{in.front(), in.substr(1)}};
            return std::nullopt;
        };
    }

    // parse character ∉ chars
    constexpr parser<char> auto none_of(std::string_view chars)
    {
        return [=] (parse_input_t in) -> parse_result_t<char> {
            if (in.empty())
                return std::nullopt;
            if (chars.find(in.front()) == chars.npos)
                return {{in.front(), in.substr(1)}};
            return std::nullopt;
        };
    }

    // accumulate parsed values by parser p1, that are separated by
    // strings parseble by p2
    template< typename P1, typename P2, typename T, typename F >
    constexpr parser<T> auto separated(P1 &&p1, P2 &&p2, T &&init, F &&f)
    {
        return [p1 = std::forward< P1 >(p1), p2 = std::forward< P2 >(p2),
                init = std::forward< T >(init), f = std::forward< F >(f)]
                (parse_input_t s)
            -> parse_result_t<T>
        {
            if (auto r = p1(s)) {
                auto p = p2 < p1;
                return {accumulate(rest(r), p, f(std::move(init), std::move(result(r))), f)};
            }
            return {{init, s}};
        };
    }

    // --- parser constructor helpers ---

    template< typename T, typename P >
    constexpr parser<T> auto from_tuple(P &&p)
    {
        return fmap([] (auto &&t) {
            return std::make_from_tuple< T >(std::forward<decltype(t)>(t));
        }, std::forward<P>(p));
    }

    template< typename T, typename P >
    constexpr parser<T> auto construct(P &&p)
    {
        return fmap([] (auto &&arg) { return T{std::forward<decltype(arg)>(arg)}; }, std::forward<P>(p) );
    }

    template< typename E, typename P >
    constexpr parser<E> auto as_enum(E value, P &&p)
    {
        return fmap([value] (auto &&) { return value; }, std::forward<P>(p) );
    }

} // namespace vast
