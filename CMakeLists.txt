cmake_minimum_required(VERSION 4.0)
project(pseudojev C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(PJEV_BUILD_TOOLS "Build the exemplar database tool (pjev-filldb)" ON)
include(CTest)  # BUILD_TESTING (default ON): unit and integration tests

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(pjev-httplib STATIC
    llama.cpp/vendor/cpp-httplib/httplib.cpp
)
target_include_directories(pjev-httplib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/vendor/cpp-httplib
)
find_package(Threads REQUIRED)
target_link_libraries(pjev-httplib PUBLIC Threads::Threads)
target_compile_features(pjev-httplib PRIVATE cxx_std_20)
if (MSVC)
    target_compile_options(pjev-httplib PRIVATE /w)
else()
    target_compile_options(pjev-httplib PRIVATE -w)
endif()
if (WIN32)
    target_link_libraries(pjev-httplib PUBLIC ws2_32)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/bin")

# Embed git commit and llama.cpp revision as compile-time strings
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE PJEV_GIT_COMMIT_VAL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if (NOT PJEV_GIT_COMMIT_VAL)
    set(PJEV_GIT_COMMIT_VAL "unknown")
endif()
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp"
    OUTPUT_VARIABLE PJEV_LLAMA_REV_VAL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if (NOT PJEV_LLAMA_REV_VAL)
    set(PJEV_LLAMA_REV_VAL "unknown")
endif()
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PJEV_VERSION_RAW)
string(STRIP "${PJEV_VERSION_RAW}" PJEV_VERSION)

# Platform/arch detection (must be before target_compile_definitions)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(PJEV_PLATFORM "windows")
    set(PJEV_ARCH "x86_64")
    set(CPACK_GENERATOR "ZIP")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(PJEV_PLATFORM "macos")
    execute_process(COMMAND uname -m OUTPUT_VARIABLE PJEV_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CPACK_GENERATOR "TGZ")
else()
    set(PJEV_PLATFORM "linux")
    execute_process(COMMAND uname -m OUTPUT_VARIABLE PJEV_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CPACK_GENERATOR "TGZ")
endif()

set(PJEV_BUILD_DEFINITIONS
    PJEV_VERSION="${PJEV_VERSION}"
    PJEV_GIT_COMMIT="${PJEV_GIT_COMMIT_VAL}"
    PJEV_LLAMA_REVISION="${PJEV_LLAMA_REV_VAL}"
    PJEV_PLATFORM="${PJEV_PLATFORM}"
    PJEV_ARCH="${PJEV_ARCH}"
)

# ---------------------------------------------------------------------------
# pjev-core: decision engine, task routing, exemplar retrieval, API, CLI formatting, worker protocol,
# configuration. No llama.cpp dependency, so unit tests link it with mock backends.
# ---------------------------------------------------------------------------
add_library(pjev-core STATIC
    src/util/platform.cpp
    src/calibration/calibration.cpp
    src/decision/prompt_strategy.cpp
    src/decision/decision_engine.cpp
    src/retrieval/embedding_text.cpp
    src/retrieval/exemplar_allowlist.cpp
    src/retrieval/exemplar_store.cpp
    src/retrieval/lz4_utils.cpp
    src/retrieval/vec_autoext.cpp
    src/routing/knn.cpp
    src/routing/task_mapping.cpp
    src/routing/task_router.cpp
    src/server/jev_api.cpp
    src/model/model_identity.cpp
    src/distribution/release_config.cpp
    src/cli/cli.cpp
    src/worker/ipc_protocol.cpp
    src/worker/runtime_dir.cpp
    src/worker/local_transport.cpp
    src/murmur3/murmur3.c
    src/sqlite/sqlite3.c
    src/sqlite/sqlite-vec.c
    src/lz4/lz4.c
    src/lz4/lz4file.c
    src/lz4/lz4frame.c
    src/lz4/lz4hc.c
    src/lz4/xxhash.c
)
target_include_directories(pjev-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/spdlog/include
)
target_compile_features(pjev-core PUBLIC cxx_std_20)
if (MSVC)
    target_compile_options(pjev-core PUBLIC $<$<COMPILE_LANGUAGE:CXX,C>:/utf-8>)
endif()
if (WIN32)
    target_link_libraries(pjev-core PUBLIC psapi)
endif()

# ---------------------------------------------------------------------------
# pjev-runtime: model loading (llama.cpp), the background worker and the HTTP server.
# ---------------------------------------------------------------------------
add_library(pjev-runtime STATIC
    src/inference/llama_backend.cpp
    src/inference/llama_instance.cpp
    src/embedding/embedding.cpp
    src/routing/router_runtime.cpp
    src/server/http_server.cpp
    src/worker/worker_server.cpp
    src/worker/worker_client.cpp
)
target_include_directories(pjev-runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)
target_compile_definitions(pjev-runtime PRIVATE ${PJEV_BUILD_DEFINITIONS})
target_link_libraries(pjev-runtime PUBLIC pjev-core llama pjev-httplib)

# ---------------------------------------------------------------------------
# pjev executable
# ---------------------------------------------------------------------------
add_executable(pjev src/main.cpp)
target_compile_definitions(pjev PRIVATE ${PJEV_BUILD_DEFINITIONS})
target_link_libraries(pjev PRIVATE pjev-runtime)
set_target_properties(pjev PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

install(TARGETS pjev RUNTIME DESTINATION . COMPONENT runtime)
install(FILES
    LICENSE
    README.md
    release/pjev.json
    release/THIRD-PARTY-NOTICES.md
    DESTINATION . COMPONENT runtime)

# CPack — release archive generation
set(CPACK_PACKAGE_NAME "pjev")
set(CPACK_PACKAGE_VERSION "${PJEV_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "pjev-${PJEV_VERSION}-${PJEV_PLATFORM}-${PJEV_ARCH}")
set(CPACK_STRIP_FILES TRUE)
# Only the runtime component: llama.cpp's own install rules (headers, static libraries) stay out.
set(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};${CMAKE_PROJECT_NAME};runtime;/")
include(CPack)

if (PJEV_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
