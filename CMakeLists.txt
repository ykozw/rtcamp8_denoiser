cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
enable_testing()
include(FetchContent)

# ソリューションエクスプローラーにcmake用のフォルダを追加する
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")

# ソリューション全体設定
project(fluorite VERSION 1.0)
set(CMAKE_CXX_STANDARD 20)

# ヘッダーオンリーライブラリ
include_directories(vendor/glm/)
include_directories(vendor/stb)
include_directories(vendor/eigen)

# TBBを配置
FetchContent_Declare(
    tbb
    URL https://github.com/oneapi-src/oneTBB/releases/download/v2021.5.0/oneapi-tbb-2021.5.0-win.zip
    )
FetchContent_MakeAvailable(tbb)
include_directories(${tbb_SOURCE_DIR}/include)
link_directories(${tbb_SOURCE_DIR}//lib/intel64/vc14)

# 外部ライブラリのインストール
INSTALL(
    FILES
        ${tbb_SOURCE_DIR}/redist/intel64/vc14/tbb12.dll
    DESTINATION ${CMAKE_SOURCE_DIR}/bin
)

# denoiserプロジェクト
project(denoiser CXX)

# static リンク
if(MSVC)
    string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_DEBUG            ${CMAKE_C_FLAGS_DEBUG})
    string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_MINSIZEREL       ${CMAKE_C_FLAGS_MINSIZEREL})
    string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELEASE          ${CMAKE_C_FLAGS_RELEASE})
    string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELWITHDEBINFO   ${CMAKE_C_FLAGS_RELWITHDEBINFO})
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_DEBUG          ${CMAKE_CXX_FLAGS_DEBUG})
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_MINSIZEREL     ${CMAKE_CXX_FLAGS_MINSIZEREL})
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE        ${CMAKE_CXX_FLAGS_RELEASE})
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
endif(MSVC)

add_executable(denoiser main.cpp)

target_compile_definitions(denoiser PRIVATE
    # GLM
    GLM_FORCE_SWIZZLE
    #GLM_FORCE_AVX
    #GLM_FORCE_MESSAGES
    # CRT
    _CRT_SECURE_NO_WARNINGS
)
target_link_libraries(denoiser PRIVATE tbb)
target_compile_options(denoiser PRIVATE
        /MP
        /arch:AVX2
        /fp:fast
        /bigobj)

INSTALL(
    TARGETS denoiser
    DESTINATION ${CMAKE_SOURCE_DIR}/bin
)
