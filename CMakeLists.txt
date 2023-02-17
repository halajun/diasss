cmake_minimum_required(VERSION 2.8)
project(diasss)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)
set(UTILITIES_DIRS ${PROJECT_SOURCE_DIR}/src/util)
set(CORE_DIRS ${PROJECT_SOURCE_DIR}/src/core)
set(THIRDPARTY_DIRS ${PROJECT_SOURCE_DIR}/thirdparty)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O3")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(PCL REQUIRED)
find_package(Boost COMPONENTS system filesystem date_time thread REQUIRED)
find_package(GTSAM REQUIRED)
if(GTSAM_FOUND)  
  message(STATUS "Found GTSAM ${GTSAM_VERSION}")
else()
  message(STATUS "GTSAM Not Found")
endif()

# find_package(LibXTF REQUIRED)
# find_package(LibGSF REQUIRED)
# find_package(EigenCereal REQUIRED)
# find_package(DataTools REQUIRED)

# Sources files
set(diasss_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/util/util.cpp
  ${PROJECT_SOURCE_DIR}/src/core/frame.cpp
  ${PROJECT_SOURCE_DIR}/src/core/FEAmatcher.cpp
  ${PROJECT_SOURCE_DIR}/thirdparty/ORBextractor.cpp
  ${PROJECT_SOURCE_DIR}/src/core/optimizer.cpp
  ${PROJECT_SOURCE_DIR}/src/core/LMtriangulatefactor.cpp
  ${PROJECT_SOURCE_DIR}/src/core/SSSpointfactor.cpp
)

add_library(diasss ${diasss_SOURCE_FILES})

target_link_libraries(diasss gtsam)

include_directories(
	${UTILITIES_DIRS}
	${CORE_DIRS}
	${THIRDPARTY_DIRS}
	${Boost_INCLUDE_DIRS}
	${EIGEN_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR} 
	# ${PCL_INCLUDE_DIRS}
	)

link_directories(
	${Boost_LIBRARY_DIRS}
	${GTSAM_LIBRARY_DIRS}
	# ${PCL_LIBRARY_DIRS}
	)

# add_definitions(${PCL_DEFINITIONS})

# add_executable(test_demo 
#                src/diasss.cpp)
add_executable(test_demo 
               src/diasss2.cpp)

target_link_libraries(test_demo diasss
		${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
# target_link_libraries(test_demo
#                       diasss gsf_data xtf_data csv_data navi_data all_data 
# 					  ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${Boost_LIBRARIES})
