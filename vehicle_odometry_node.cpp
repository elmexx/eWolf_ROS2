#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "vehicle_dynamic_msg/msg/vehicle_dynamic.hpp"

class VehicleDynamicToOdometry : public rclcpp::Node
{
public:
    VehicleDynamicToOdometry() : Node("vehicle_dynamic_to_odometry")
    {
        subscription_ = this->create_subscription<vehicle_dynamic_msg::msg::VehicleDynamic>(
            "/vehicle_dynamic_data", 10,
            std::bind(&VehicleDynamicToOdometry::callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry/vehicle", 10);
    }

private:
    void callback(const vehicle_dynamic_msg::msg::VehicleDynamic::SharedPtr msg)
    {
        auto odometry_msg = nav_msgs::msg::Odometry();
        odometry_msg.header.stamp = msg->header.stamp;
        odometry_msg.header.frame_id = "odom";
        odometry_msg.child_frame_id = "base_link";

        // Set linear velocity from car_speed
        odometry_msg.twist.twist.linear.x = msg->car_speed;

        // Set angular velocity from yaw_rate
        odometry_msg.twist.twist.angular.z = msg->yaw_rate;

        publisher_->publish(odometry_msg);
    }

    rclcpp::Subscription<vehicle_dynamic_msg::msg::VehicleDynamic>::SharedPtr subscription_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VehicleDynamicToOdometry>());
    rclcpp::shutdown();
    return 0;
}
