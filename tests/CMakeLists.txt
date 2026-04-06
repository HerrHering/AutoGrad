cmake_minimum_required(VERSION 3.16)
project(MyAutoGradTests CXX)

# 1. Download Catch2 v3 automatically
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.4.0 # Or latest version
)

# Get Eigen from parent scope (already fetched)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Catch2 Eigen)

# Get Eigen source directory for include path
FetchContent_GetProperties(Eigen SOURCE_DIR EIGEN_SOURCE_DIR)

# 2. Grab your source files for testing
# We reach up to the parent directory to find your matrix implementation
file(GLOB_RECURSE TEST_SOURCES "test_*.cpp")
#file(GLOB_RECURSE PROJECT_SOURCES "../src/*.cpp")

# 3. Create the test executable
add_executable(unit_tests ${TEST_SOURCES})

# 4. Set include paths so tests can find your headers
target_include_directories(unit_tests PRIVATE ../include)
target_include_directories(unit_tests PUBLIC ${EIGEN_SOURCE_DIR})

# 5. Link against Catch2's built-in main
# Catch2::Catch2WithMain handles the main() function for you!
target_link_libraries(unit_tests PRIVATE Catch2::Catch2WithMain)

# 6. Enable C++20 for tests
target_compile_features(unit_tests PRIVATE cxx_std_20)

include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
catch_discover_tests(unit_tests PROPERTIES ARGS "-s -r compact")