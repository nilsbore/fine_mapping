cmake_minimum_required(VERSION 2.8.3)
project(mapping_refinement)

add_definitions(-std=c++0x -O4 -DNDEBUG -fPIC)
#set(CMAKE_CXX_FLAGS "-DCMAKE_BUILD_TYPE=Debug -Wall -fPIC -std=c++0x")
set(CATKIN_DEVEL_PREFIX "/home/nbore/hydro_ws2/devel")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS std_msgs roscpp sensor_msgs std_msgs pcl_ros tf message_filters cv_bridge qt_build metaroom_xml_parser)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

find_package(OpenCV REQUIRED)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

rosbuild_prepare_qt4(QtCore QtXml)

#include_directories("/home/nbore/Installs/opencv/include")
#link_directories("/home/nbore/Installs/opencv/release/lib")

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/groovy/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
add_service_files(
  FILES
  SaveScan.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
   INCLUDE_DIRS include
   LIBRARIES scan fine_registration asynch_visualizer stitched_map octave_convenience fine_mapping
#  CATKIN_DEPENDS other_catkin_pkg
#   DEPENDS
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include /opt/ros/hydro/include/ ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
link_directories(/opt/ros/hydro/lib/)

## Declare a cpp library
add_library(scan src/scan.cpp include/scan.h)
add_library(fine_registration src/fine_registration.cpp include/fine_registration.h)
add_library(asynch_visualizer src/asynch_visualizer.cpp include/asynch_visualizer.h)
add_library(stitched_map src/stitched_map.cpp include/stitched_map.h)
add_library(octave_convenience src/octave_convenience.cpp include/octave_convenience.h)
add_library(fine_mapping src/fine_mapping.cpp include/fine_mapping.h)

## Declare a cpp executable
add_executable(save_scan src/save_scan.cpp)
add_executable(visualize_registration src/visualize_registration.cpp)
add_executable(test_cuda src/test_cuda.cpp)
add_executable(test_mapping src/test_mapping.cpp)
add_executable(test_icp_mapping src/test_icp_mapping.cpp)
add_executable(compare_registration src/compare_registration.cpp)
add_executable(reproject_scans src/reproject_scans.cpp)
add_executable(map_from_scan src/map_from_scan.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
add_dependencies(save_scan mapping_refinement_generate_messages_cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(save_scan
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  #opencv2_all
)

target_link_libraries(scan
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  octave_convenience
)

target_link_libraries(fine_registration
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  #opencv2_all
  scan
)

target_link_libraries(asynch_visualizer
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

target_link_libraries(stitched_map
  ${catkin_LIBRARIES}
  scan
  ${OpenCV_LIBS}
  #opencv2_all
)


target_link_libraries(visualize_registration
  ${catkin_LIBRARIES}
  scan
  fine_registration
  asynch_visualizer
)

target_link_libraries(test_cuda
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  #opencv2_all
)

target_link_libraries(test_icp_mapping
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  fine_registration
  asynch_visualizer
  g2o_core
  g2o_types_slam3d
  g2o_solver_csparse
  g2o_stuff
)

target_link_libraries(compare_registration
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  fine_registration
  asynch_visualizer
  g2o_core
  g2o_types_slam3d
  g2o_solver_csparse
  g2o_stuff
)

target_link_libraries(reproject_scans
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  scan
  fine_registration
  asynch_visualizer
)

target_link_libraries(fine_mapping
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  fine_registration
  stitched_map
  scan
  g2o_core
  g2o_types_slam3d
  g2o_solver_csparse
  g2o_stuff
)

target_link_libraries(map_from_scan
  scan
  fine_mapping
  stitched_map
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

target_link_libraries(test_mapping
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  scan
  fine_mapping
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/groovy/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS strands_morse strands_morse_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_strands_morse.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
