#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "vehicle_dynamic_msg/msg/vehicle_dynamic.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cmath>

class VehicleDynamicToOdometry : public rclcpp::Node
{
public:
    VehicleDynamicToOdometry()
        : Node("vehicle_dynamic_to_odometry"),
          last_time_(0, 0, RCL_ROS_TIME) 
    {
        
        subscription_ = this->create_subscription<vehicle_dynamic_msg::msg::VehicleDynamic>(
            "/vehicle_dynamic_data", 10,
            std::bind(&VehicleDynamicToOdometry::callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry/vehicle", 10);

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }

private:
    void callback(const vehicle_dynamic_msg::msg::VehicleDynamic::SharedPtr msg)
    {

        rclcpp::Time current_time = msg->header.stamp;

        double dt = 0.0;
        if (last_time_.nanoseconds() > 0) 
        {
            dt = (current_time - last_time_).seconds();
        }
        last_time_ = current_time;

        if (dt <= 0.0)
        {
            RCLCPP_WARN(this->get_logger(), "Invalid dt detected, skipping this callback.");
            return;
        }

        auto odometry_msg = nav_msgs::msg::Odometry();
        odometry_msg.header.stamp = msg->header.stamp;
        odometry_msg.header.frame_id = "odom";
        odometry_msg.child_frame_id = "base_link";

        static double x = 0.0, y = 0.0, theta = 0.0;

        x += msg->car_speed * dt * cos(theta);
        y += msg->car_speed * dt * sin(theta);
        theta += msg->yaw_rate * dt;

        odometry_msg.pose.pose.position.x = x;
        odometry_msg.pose.pose.position.y = y;
        odometry_msg.pose.pose.position.z = 0.0;

        odometry_msg.pose.pose.orientation.z = sin(theta / 2.0);
        odometry_msg.pose.pose.orientation.w = cos(theta / 2.0);

        odometry_msg.twist.twist.linear.x = msg->car_speed;
        odometry_msg.twist.twist.angular.z = msg->yaw_rate;

        publisher_->publish(odometry_msg);

        publish_tf(msg->header.stamp, x, y, theta);
    }

    void publish_tf(const rclcpp::Time &stamp, double x, double y, double theta)
    {
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = stamp;
        transform.header.frame_id = "odom";
        transform.child_frame_id = "base_link";

        transform.transform.translation.x = x;
        transform.transform.translation.y = y;
        transform.transform.translation.z = 0.0;

        transform.transform.rotation.x = 0.0;
        transform.transform.rotation.y = 0.0;
        transform.transform.rotation.z = sin(theta / 2.0);
        transform.transform.rotation.w = cos(theta / 2.0);

        tf_broadcaster_->sendTransform(transform);
    }

    rclcpp::Subscription<vehicle_dynamic_msg::msg::VehicleDynamic>::SharedPtr subscription_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_; 

    rclcpp::Time last_time_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VehicleDynamicToOdometry>());
    rclcpp::shutdown();
    return 0;
}
