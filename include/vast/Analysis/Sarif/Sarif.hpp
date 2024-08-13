// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#ifdef VAST_ENABLE_SARIF
    #include <vector>

    #include <gap/sarif/sarif.hpp>

namespace vast::analysis::sarif {

    class sarif_analysis
    {
      private:
        std::vector< gap::sarif::result > sarif_results;

      public:
        virtual ~sarif_analysis() = default;

        const std::vector< gap::sarif::result > &results() const noexcept {
            return sarif_results;
        }

      protected:
        void append_result(const gap::sarif::result &result) {
            sarif_results.push_back(result);
        }

        void append_result(gap::sarif::result &&result) {
            sarif_results.emplace_back(std::move(result));
        }
    };

} // namespace vast::analysis::sarif
#endif // VAST_ENABLE_SARIF
