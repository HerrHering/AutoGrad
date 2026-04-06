cmake_minimum_required(VERSION 3.16)

project(MyAutoGrad CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Fetch and Build Eigen3
# -----------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(Eigen)

# Get Eigen source directory for include path
FetchContent_GetProperties(Eigen SOURCE_DIR EIGEN_SOURCE_DIR)

# -----------------------------------------------------------------------------
# Project Sources and Executable
# -----------------------------------------------------------------------------

# Grab source files
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "include/*.hpp")

# Define our executable target first
set(EXE_NAME myautograd)
add_executable(${EXE_NAME} ${SOURCE_FILES})

# ---------------------------------------------
# Set include path for headers
target_include_directories(${EXE_NAME} PUBLIC include)
# Add Eigen include directory
target_include_directories(${EXE_NAME} PUBLIC ${EIGEN_SOURCE_DIR})
# Link Eigen library if available, otherwise just include path
if(TARGET Eigen3::Eigen)
    target_link_libraries(${EXE_NAME} PUBLIC Eigen3::Eigen)
endif()
# ---------------------------------------------

add_custom_target(copy_assets ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets"
            "${CMAKE_CURRENT_BINARY_DIR}/assets"
    COMMENT "Copying assets to build directory"
)
add_dependencies(${EXE_NAME} copy_assets)

# # Create a custom target that copies the shaders.
# # The "ALL" keyword ensures this target is run as part of the default build.
# add_custom_target(copy_assets ALL
#     COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
#             "${CMAKE_SOURCE_DIR}/assets"
#             "${CMAKE_CURRENT_BINARY_DIR}/assets"
#     COMMENT "Copying assets to build directory"
# )
# # Make our main executable depend on the copy_assets target.
# # This guarantees assets are copied before the executable is built.
# add_dependencies(raylib_test_project copy_assets)

# -----------------------------------------------------------------------------
# Apply Quality of Life Compiler Flags ONLY to our own target
# -----------------------------------------------------------------------------
if(MSVC)
    # Warning level 4, treat warnings as errors
    target_compile_options(${EXE_NAME} PRIVATE /W4 /WX)
    # Enable AVX2 for MSVC
    target_compile_options(${EXE_NAME} PRIVATE /arch:AVX2)
else()
    # Enable a good set of warnings and treat them as errors for our code
    target_compile_options(${EXE_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    
    # Add debug information for our code
    target_compile_options(${EXE_NAME} PRIVATE -g)
    
    # Enable CPU-specific optimizations: AVX2 for broad compatibility, or -march=native for maximum performance
    # Option 1: AVX2 (recommended - good balance, works on most modern CPUs)
    # target_compile_options(${EXE_NAME} PRIVATE -mavx2)
    
    # Option 2: Uncomment below for maximum CPU-specific optimization (AVX-512 on your machine)
    target_compile_options(${EXE_NAME} PRIVATE -march=native)
endif()

# -----------------------------------------------------------------------------
# Link against raylib
# -----------------------------------------------------------------------------
# target_link_libraries(raylib_test_project PRIVATE raylib)

# # Include raylib headers for our target
# target_include_directories(raylib_test_project PRIVATE ${raylib_SOURCE_DIR}/src)

include(CTest)
enable_testing()
add_subdirectory(tests)