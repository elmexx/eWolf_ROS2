cmake_minimum_required(VERSION 3.5)
project(lane_detection_visualization)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(lane_parameter_msg REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

include_directories(
        include
)

add_executable(lane_detection_visualization_node src/lane_detection_visualization.cpp)
ament_target_dependencies(lane_detection_visualization_node rclcpp lane_parameter_msg visualization_msgs geometry_msgs std_msgs ament_index_cpp Boost)
target_link_libraries(lane_detection_visualization_node
  ${rclcpp_LIBRARIES}
  ${Boost_LIBRARIES}
  ${visualization_msgs_LIBRARIES}
  ${ament_index_cpp_LIBRARIES}
  dl -lboost_system -lm -lz -lpthread
)

install(
  TARGETS lane_detection_visualization_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY resources
  DESTINATION share/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
