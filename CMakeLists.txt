cmake_minimum_required(VERSION 3.16)

project(MyAutoGrad CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Fetch and Build raylib
# -----------------------------------------------------------------------------
# include(FetchContent)

# FetchContent_Declare(
#   raylib
#   GIT_REPOSITORY https://github.com/raysan5/raylib.git
#   GIT_TAG master # You can pin this to a specific version if you prefer
# )

# FetchContent_MakeAvailable(raylib)

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
# ---------------------------------------------

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
else()
    # Enable a good set of warnings and treat them as errors for our code
    target_compile_options(${EXE_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    
    # Add debug information for our code
    target_compile_options(${EXE_NAME} PRIVATE -g)
endif()

# -----------------------------------------------------------------------------
# Link against raylib
# -----------------------------------------------------------------------------
# target_link_libraries(raylib_test_project PRIVATE raylib)

# # Include raylib headers for our target
# target_include_directories(raylib_test_project PRIVATE ${raylib_SOURCE_DIR}/src)