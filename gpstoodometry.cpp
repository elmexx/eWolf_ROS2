#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cmath>

class GpsToOdometry : public rclcpp::Node
{
public:
    GpsToOdometry()
        : Node("gps_to_odometry_node"),
          ref_lat_(0.0), ref_lon_(0.0), ref_set_(false)
    {
        gps_subscription_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/fix", 10, std::bind(&GpsToOdometry::gps_callback, this, std::placeholders::_1));
        odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry/gps", 10);
    }

private:
    void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
        if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX)
        {
            RCLCPP_WARN(this->get_logger(), "No valid GPS fix.");
            return;
        }

        if (!ref_set_)
        {
            // Set reference point
            ref_lat_ = msg->latitude;
            ref_lon_ = msg->longitude;
            ref_alt_ = msg->altitude;
            ref_set_ = true;
            RCLCPP_INFO(this->get_logger(), "Reference point set: lat=%f, lon=%f, alt=%f", ref_lat_, ref_lon_, ref_alt_);
        }

        // Convert GPS to local ENU (East-North-Up) coordinates
        double x, y, z;
        gps_to_local(msg->latitude, msg->longitude, msg->altitude, x, y, z);

        // Create Odometry message
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = msg->header.stamp;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";

        // Fill in pose
        odom_msg.pose.pose.position.x = x;
        odom_msg.pose.pose.position.y = y;
        odom_msg.pose.pose.position.z = z;

        // Publish the Odometry message
        odometry_publisher_->publish(odom_msg);
    }

    void gps_to_local(double lat, double lon, double alt, double &x, double &y, double &z)
    {
        const double earth_radius = 6378137.0; // Earth radius in meters

        // Convert lat/lon to radians
        double lat_rad = lat * M_PI / 180.0;
        double lon_rad = lon * M_PI / 180.0;
        double ref_lat_rad = ref_lat_ * M_PI / 180.0;
        double ref_lon_rad = ref_lon_ * M_PI / 180.0;

        // Calculate East-North-Up (ENU) coordinates
        double delta_lat = lat_rad - ref_lat_rad;
        double delta_lon = lon_rad - ref_lon_rad;

        x = earth_radius * delta_lon * cos(ref_lat_rad); // East
        y = earth_radius * delta_lat;                   // North
        z = alt - ref_alt_;                             // Up
    }

    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_subscription_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;

    double ref_lat_, ref_lon_, ref_alt_;
    bool ref_set_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GpsToOdometry>());
    rclcpp::shutdown();
    return 0;
}


cmake_minimum_required(VERSION 3.8)
project(navsat_to_odometry)

# Ensure you include the C++ standard version
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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


<?xml version="1.0"?>
<package format="2">
  <name>navsat_to_odometry</name>
  <version>0.1.0</version>
  <description>Package to convert NavSatFix messages to Odometry messages</description>
  <maintainer email="your_email@example.com">Your Name</maintainer>
  <license>Apache 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <!-- Dependencies -->
  <depend>rclcpp</depend>
  <depend>sensor_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>tf2</depend>
  <depend>tf2_geometry_msgs</depend>
  <depend>tf2_ros</depend>

  <!-- Optional testing dependencies -->
  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_cmake_pytest</test_depend>

  <!-- Export the buildtool -->
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>

