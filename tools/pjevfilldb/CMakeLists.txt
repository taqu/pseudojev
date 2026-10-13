cmake_minimum_required(VERSION 3.14)

if(MSVC)
    add_compile_options(/utf-8)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG         "${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE       "${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO"${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL    "${PROJECT_ROOT}/bin")

set(PJEV_FILLDB_INCS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/vendor
    ${CMAKE_SOURCE_DIR}/llama.cpp/vendor/cpp-httplib
    ${CMAKE_SOURCE_DIR}/spdlog/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(pjev-filldb
    main.cpp
    exemplar_record.cpp
    embedding_text.cpp
    exemplar_database.cpp
    lz4_utils.cpp
    vec_autoext.cpp
    ${CMAKE_SOURCE_DIR}/src/embedding/embedding.cpp
    ${CMAKE_SOURCE_DIR}/src/inference/llama_instance.cpp
    ${CMAKE_SOURCE_DIR}/src/util/platform.cpp
    ${CMAKE_SOURCE_DIR}/src/model/model_identity.cpp
    ${CMAKE_SOURCE_DIR}/src/sqlite/sqlite3.c
    ${CMAKE_SOURCE_DIR}/src/sqlite/sqlite-vec.c
    ${CMAKE_SOURCE_DIR}/src/lz4/lz4.c
    ${CMAKE_SOURCE_DIR}/src/lz4/lz4hc.c
    ${CMAKE_SOURCE_DIR}/src/lz4/lz4frame.c
    ${CMAKE_SOURCE_DIR}/src/lz4/xxhash.c
)
target_include_directories(pjev-filldb PRIVATE ${PJEV_FILLDB_INCS})
target_link_libraries(pjev-filldb PRIVATE llama pjev-httplib)
if(WIN32)
    target_link_libraries(pjev-filldb PRIVATE psapi)
endif()
set_target_properties(pjev-filldb PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_ROOT}")
target_compile_features(pjev-filldb PRIVATE cxx_std_20)
