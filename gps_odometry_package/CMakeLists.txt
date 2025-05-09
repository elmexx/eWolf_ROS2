cmake_minimum_required(VERSION 3.5)
project(gps_odometry_package)

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

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# Add the executable
add_executable(navsat_to_odom src/navsat_to_odom.cpp)

# Link the required libraries
ament_target_dependencies(
  navsat_to_odom
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

# Install the executable
install(TARGETS
  navsat_to_odom
  DESTINATION lib/${PROJECT_NAME})

# Install other files if needed (e.g., launch files)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Package export
ament_package()
