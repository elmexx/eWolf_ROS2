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
          last_time_(0, 0, RCL_ROS_TIME) // 初始化时间戳
    {
        // 订阅车辆动态数据
        subscription_ = this->create_subscription<vehicle_dynamic_msg::msg::VehicleDynamic>(
            "/vehicle_dynamic_data", 10,
            std::bind(&VehicleDynamicToOdometry::callback, this, std::placeholders::_1));

        // 发布车辆里程计数据
        publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry/vehicle", 10);

        // 初始化 TF 广播器
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }

private:
    void callback(const vehicle_dynamic_msg::msg::VehicleDynamic::SharedPtr msg)
    {
        // 当前消息的时间戳
        rclcpp::Time current_time = msg->header.stamp;

        // 动态计算 dt
        double dt = 0.0;
        if (last_time_.nanoseconds() > 0) // 确保有上次时间的有效值
        {
            dt = (current_time - last_time_).seconds();
        }
        last_time_ = current_time;

        // 如果 dt 太小或无效，跳过计算
        if (dt <= 0.0)
        {
            RCLCPP_WARN(this->get_logger(), "Invalid dt detected, skipping this callback.");
            return;
        }

        // 创建 Odometry 消息
        auto odometry_msg = nav_msgs::msg::Odometry();
        odometry_msg.header.stamp = msg->header.stamp;
        odometry_msg.header.frame_id = "odom";
        odometry_msg.child_frame_id = "base_link";

        // 静态变量存储累计的位置和姿态
        static double x = 0.0, y = 0.0, theta = 0.0;

        // 根据车速和偏航角速度计算位置和姿态
        x += msg->car_speed * dt * cos(theta);
        y += msg->car_speed * dt * sin(theta);
        theta += msg->yaw_rate * dt;

        // 设置位置
        odometry_msg.pose.pose.position.x = x;
        odometry_msg.pose.pose.position.y = y;
        odometry_msg.pose.pose.position.z = 0.0;

        // 设置姿态（转换为四元数）
        odometry_msg.pose.pose.orientation.z = sin(theta / 2.0);
        odometry_msg.pose.pose.orientation.w = cos(theta / 2.0);

        // 设置速度
        odometry_msg.twist.twist.linear.x = msg->car_speed;
        odometry_msg.twist.twist.angular.z = msg->yaw_rate;

        // 发布 Odometry 消息
        publisher_->publish(odometry_msg);

        // 发布 TF (odom -> base_link)
        publish_tf(msg->header.stamp, x, y, theta);
    }

    void publish_tf(const rclcpp::Time &stamp, double x, double y, double theta)
    {
        // 创建 TransformStamped 消息
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = stamp;
        transform.header.frame_id = "odom";
        transform.child_frame_id = "base_link";

        // 设置平移
        transform.transform.translation.x = x;
        transform.transform.translation.y = y;
        transform.transform.translation.z = 0.0;

        // 设置旋转（转换为四元数）
        transform.transform.rotation.x = 0.0;
        transform.transform.rotation.y = 0.0;
        transform.transform.rotation.z = sin(theta / 2.0);
        transform.transform.rotation.w = cos(theta / 2.0);

        // 广播 TF
        tf_broadcaster_->sendTransform(transform);
    }

    rclcpp::Subscription<vehicle_dynamic_msg::msg::VehicleDynamic>::SharedPtr subscription_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_; // 动态 TF 广播器

    rclcpp::Time last_time_; // 上次消息的时间戳
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VehicleDynamicToOdometry>());
    rclcpp::shutdown();
    return 0;
}
