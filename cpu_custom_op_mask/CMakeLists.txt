# cmake_minimum_required(VERSION 3.16)
# project(custom_layer)
# set(CMAKE_CXX_STANDARD 11)

# set(TARGET_NAME "arsenal")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -std=gnu++2a -mavx512f -mavx512vl -mavx512pf -mavx512er -mavx512cd -fpermissive -O3 -std=c++11")
# message(STATUS "Enabling AVX512")

# # Sets variable to a value if variable is unset.
# macro(set_ifndef var val)
#     if (NOT ${var})
#         set(${var} ${val})
#     endif ()
#     message(STATUS "Configurable variable ${var} set to ${${var}}")
# endmacro()

# set_ifndef(_COMMMON_LIB /usr/lib/x86_64-linux-gnu)

# find_package(REQUIRED OPTIONAL_COMPONENTS onnx_importer)
# find_package(InferenceEngine REQUIRED)
# find_package(OpenCV QUIET COMPONENTS core)


# add_library(${TARGET_NAME} MODULE 

# 	${CMAKE_SOURCE_DIR}/ov_extension.cpp

# 	${CMAKE_SOURCE_DIR}/simplegpu/simplegpu.cpp
#         ${CMAKE_SOURCE_DIR}/simplegpu/simplegpu.hpp
# 	)



# target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
# target_link_libraries(${TARGET_NAME} PRIVATE IE::inference_engine
#                                              ${NGRAPH_LIBRARIES}
# 					     #/data/zhaohb/framework/openvino_dir/install_2022_3/runtime/3rdparty/tbb/lib/libtbb.so)
# 					     #/opt/intel/openvino/runtime/3rdparty/tbb/lib/libtbb.so)
# 					     )

# if (ngraph_onnx_importer_FOUND)
#     target_link_libraries(${TARGET_NAME} PRIVATE ${ONNX_IMPORTER_LIBRARIES})
#     target_compile_definitions(${TARGET_NAME} PRIVATE NGRAPH_ONNX_IMPORT_ENABLED)
# endif()

# include_directories(PRIVATE
# 	#"/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/include/"
# 	"/opt/intel/openvino/runtime/3rdparty/tbb/include/"
# 	#"/data/zhaohb/framework/openvino_dir/install_2022_3/runtime/3rdparty/tbb/include"
# 	)
# link_directories(
# 	"/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib/"
# 	#"/data/zhaohb/framework/openvino_dir/install_2022_3/runtime/3rdparty/tbb/lib"
# 	)

cmake_minimum_required(VERSION 3.22)
project(custom_layer)
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "openvino_template_extension")

# The OpenVINO installed from PyPI can be used to find OpenVINO_DIR
find_package(Python3 REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
    OUTPUT_VARIABLE OpenVINO_DIR_PY
    ERROR_QUIET
)

find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}")

# set(SRC identity.cpp ov_extension.cpp)

add_library(${TARGET_NAME} MODULE 
	${CMAKE_SOURCE_DIR}/ov_extension.cpp
	${CMAKE_SOURCE_DIR}/simplegpu/simplegpu.cpp
    ${CMAKE_SOURCE_DIR}/simplegpu/simplegpu.hpp
)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)