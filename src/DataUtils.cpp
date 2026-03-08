#include <DataUtils.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>


namespace DUtils {
    std::vector<std::string> read_cs_line(const std::string& line) {
        std::stringstream line_stream(line);
        std::vector<std::string> result;
        std::string val;
        while (std::getline(line_stream, val, ',')) {
            result.push_back(std::move(val));
        }
        return result;
    }
    SourceDataRows read_csv(const std::string &file_path, bool header)
    {
        SourceDataRows data;
        std::ifstream file(file_path);
        std::string line;
        if (header) {
            std::getline(file, line);
            data.column_names = read_cs_line(line);
        }
        while (std::getline(file, line)) {
            data.rows.push_back(read_cs_line(line));
        }
        return data;
    }
}