cmake_minimum_required(VERSION 2.8)
project(diasss)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)
set(UTILITIES_DIRS ${PROJECT_SOURCE_DIR}/src/util)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O3")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem date_time REQUIRED)

find_package(LibXTF REQUIRED)
find_package(LibGSF REQUIRED)
find_package(EigenCereal REQUIRED)
find_package(DataTools REQUIRED)

include_directories(
	${UTILITIES_DIRS}
	${Boost_INCLUDE_DIRS}
	${EIGEN_INCLUDE_DIRS}
	)

link_directories(${Boost_LIBRARY_DIRS})

add_executable(test_demo 
               src/diasss.cpp)

target_link_libraries(test_demo
                      gsf_data xtf_data csv_data navi_data all_data)
