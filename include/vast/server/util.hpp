// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace vast::server {
    template< typename Hash = std::hash< std::string > >
    class ci_hash
    {
        Hash hash;

      public:
        ci_hash(const Hash &hash = {}) : hash(hash) {}

        std::uint64_t operator()(const std::string &s) const {
            std::string lower(s.size(), '\0');
            std::transform(s.begin(), s.end(), lower.begin(), [](char c) {
                return static_cast< char >(std::tolower(static_cast< unsigned char >(c)));
            });
            return hash(lower);
        }
    };

    template< typename Comparison = std::equal_to< std::string > >
    class ci_comparison
    {
        Comparison comp;

      public:
        ci_comparison(const Comparison &comp = {}) : comp(comp) {}

        bool operator()(const std::string &a, const std::string &b) const {
            std::string a_lower(a.size(), '\0');
            std::string b_lower(b.size(), '\0');

            auto to_lower = [](char c) -> char {
                return static_cast< char >(std::tolower(static_cast< unsigned char >(c)));
            };

            std::transform(a.begin(), a.end(), a_lower.begin(), to_lower);
            std::transform(b.begin(), b.end(), b_lower.begin(), to_lower);

            return comp(a_lower, b_lower);
        }
    };
} // namespace vast::server
