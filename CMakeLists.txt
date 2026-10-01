cmake_minimum_required(VERSION 4.0)
project(pseudojev C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(pjev-httplib STATIC
    llama.cpp/vendor/cpp-httplib/httplib.cpp
)
target_include_directories(pjev-httplib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/vendor/cpp-httplib
)
find_package(Threads REQUIRED)
target_link_libraries(pjev-httplib PUBLIC Threads::Threads)
target_compile_features(pjev-httplib PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(pjev-httplib PRIVATE /w)
else()
    target_compile_options(pjev-httplib PRIVATE -w)
endif()
if (WIN32)
    target_link_libraries(pjev-httplib PUBLIC ws2_32)
endif()

set(PJEV_INCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/vendor/cpp-httplib
)

add_executable(pjev
    src/main.cpp
    src/inference/llama_backend.cpp
    src/decision/prompt_strategy.cpp
    src/decision/decision_engine.cpp
    src/server/jev_api.cpp
    src/server/http_server.cpp
    src/experiment/dataset.cpp
    src/experiment/runner.cpp
)
target_include_directories(pjev PRIVATE ${PJEV_INCS})
target_link_libraries(pjev PRIVATE llama pjev-httplib)

enable_testing()
add_subdirectory(tests)
