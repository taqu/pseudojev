cmake_minimum_required(VERSION 3.14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG         "${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE       "${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL    "${PROJECT_ROOT}/bin")

# pjev-filldb: builds the exemplar database and its allowlist (see README: Database setup).
add_library(pjev-filldb-lib STATIC
    exemplar_record.cpp
    exemplar_database.cpp
)
target_include_directories(pjev-filldb-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pjev-filldb-lib PUBLIC pjev-core)

add_executable(pjev-filldb main.cpp)
target_link_libraries(pjev-filldb PRIVATE pjev-filldb-lib pjev-runtime)
set_target_properties(pjev-filldb PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_ROOT}")
