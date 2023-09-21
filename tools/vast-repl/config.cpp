// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/repl/config.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Warnings.hpp"

#include <ranges>

namespace vast::repl::ini {

    namespace sr = std::ranges;
    namespace sv = std::views;

    struct line_proxy
    {
        friend std::istream &operator>>(std::istream &is, line_proxy &proxy) {
            std::getline(is, proxy.line);
            return is;
        }

        operator std::string() const { return line; }
        operator string_ref() const { return line; }
        std::string line{};
    };

    auto lines(std::ifstream &in) {
        return std::ranges::istream_view< line_proxy >(in);
    }

    bool section::is_sticky_section() const {
        return string_ref(name.name).starts_with("repl:boot:sticky");
    }

    bool section::is_pipeline_section() const {
        return string_ref(name.name).starts_with("repl:pipeline");
    }

    string_ref section::last_name() const {
        return string_ref(name.name).rsplit(':').second;
    }

    std::optional< section_name > section_name::parse(std::ifstream &in) {
        auto is_not_name = [] (string_ref line) {
            return !line.starts_with("[");
        };

        auto process = [] (string_ref line) -> std::optional< section_name > {
            if (!line.consume_front("[") || !line.consume_back("]")) {
                return std::nullopt;
            }

            return section_name{ line.str() };
        };

        // parse section name
        for (auto line : lines(in)) {
            // trim commented and empty lines
            if (is_not_name(line))
                continue;
            return process(line);
        }

        return std::nullopt;
    }

    void section::dump() const {
        llvm::outs() << "[" << name.name << "]\n";
        for (auto line : content) {
            llvm::outs() << line << "\n";
        }
    }

    std::optional< section > section::parse(std::ifstream &in) {
        section sec;

        if (auto name = section_name::parse(in)) {
            sec.name = name.value();
        } else {
            return std::nullopt;
        }

        auto is_not_empty = [] (string_ref line) { return !line.empty(); };

        for (std::string line : lines(in) | sv::take_while(is_not_empty)) {
            if (!line.starts_with(";")) {
                sec.content.push_back(line);
            }
        }

        return sec;
    }

    config config::parse(std::ifstream &in) {
        config cfg;
        auto next = [&] { return section::parse(in); };
        for (auto sec = next(); sec; sec = next()) {
            cfg.sections.push_back(std::move(sec.value()));
        }
        return cfg;
    }

    void config::dump() const {
        for (const auto &sec : sections) {
            sec.dump();
            llvm::outs() << "\n";
        }
    }

} // namespace vast::repl::ini
