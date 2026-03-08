#pragma once

#include <vector>
#include <string>
#include <optional>
#include <MUtils.hpp>
#include <ranges>
#include <algorithm>

namespace DUtils
{
    struct SourceDataRows {
        std::optional<std::vector<std::string>> column_names;
        std::vector<std::vector<std::string>> rows;
    };
    SourceDataRows read_csv(const std::string& file, bool header = true);

    template <typename T = float, typename D, typename Func>
    auto getColumnAs(std::vector<D>& data,  Func extractor) {
        static_assert(std::is_invocable_r_v<T&, Func, D&>, "Extractor must be callable with extractor(D&) -> T&");

        return data | std::views::transform([extractor](D& row) -> T& {
            return extractor(row);
        });
    }
    void normalize_minMaxScale(std::ranges::sized_range auto&& column) {
        static_assert(std::is_convertible_v<std::remove_cvref_t<std::ranges::range_reference_t<decltype(column)>>, float>, "normalize only accepts ranges of floats");

        if (std::ranges::empty(column)) return;

        auto result = std::ranges::minmax(column);
        float range = result.max - result.min;

        // If all the values are the same
        if (range < 1e-9f) {
            std::ranges::fill(column, 0.0f);
            return;
        }
        
        // Normalize data by: (value - min) / (max - min)
        for (auto& val : column) {
            val = (val - result.min) / range;
        }
    }
} // namespace DUtils
