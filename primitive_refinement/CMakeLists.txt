cmake_minimum_required(VERSION 2.8.3)
project(primitive_refinement)

add_definitions(-std=c++0x -O4 -DNDEBUG -fPIC)
set(CATKIN_DEVEL_PREFIX "/home/nbore/hydro_ws2/devel")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp tf ransac_primitives visualization_msgs)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

find_package(OpenCV REQUIRED)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

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
#add_service_files(
#  FILES
#  SaveScan.srv
#)

## Generate added messages and services with any dependencies listed here
#generate_messages(
#  DEPENDENCIES
#  std_msgs  # Or other packages containing msgs
#)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES strands_morse
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

set(CORE_HEADERS
include/primitive_refiner.h include/primitive_refiner.hpp include/refinement_core.h)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

## Declare a cpp library
add_library(refinement_core src/refinement_core.cpp ${CORE_HEADERS})
add_library(graph_extractor src/graph_extractor.cpp include/graph_extractor.h)
add_library(conversions src/conversions.cpp include/conversions.h)

## Declare a cpp executable
add_executable(test_refinement src/test_refinement.cpp)
add_executable(test_multiple_clouds src/test_multiple_clouds.cpp)
add_executable(split_multiple_clouds src/split_multiple_clouds.cpp)
add_executable(split_multiple_final src/split_multiple_final.cpp)
add_executable(split_multiple_final2 src/split_multiple_final2.cpp)
add_executable(pick_areas src/pick_areas.cpp)
add_executable(area_graph src/area_graph.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
#add_dependencies(save_scan mapping_refinement_generate_messages_cpp)

## Specify libraries to link a library or executable target against

target_link_libraries(refinement_core
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(graph_extractor
  ${catkin_LIBRARIES}
  conversions
)

target_link_libraries(test_refinement
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  refinement_core
)

target_link_libraries(test_multiple_clouds
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  refinement_core
  graph_extractor
)

target_link_libraries(split_multiple_clouds
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  refinement_core
  graph_extractor
)

target_link_libraries(split_multiple_final
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  refinement_core
  graph_extractor
)

target_link_libraries(split_multiple_final2
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  refinement_core
  graph_extractor
)

target_link_libraries(area_graph
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  refinement_core
  graph_extractor
)

target_link_libraries(pick_areas
  ${PCL_LIBRARIES}
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
