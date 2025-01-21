#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include "lane_parameter_msg/msg/lane_marking_projected_array_both.hpp"
#include "lane_parameter_msg/msg/lane_marking_projected.hpp"

class LaneTransformerNode : public rclcpp::Node
{
public:
    LaneTransformerNode() : Node("lane_transformer_node")
    {
        lane_markings_subscription_ = this->create_subscription<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>(
            "/lane_markings_projected", 10,
            std::bind(&LaneTransformerNode::laneMarkingsCallback, this, std::placeholders::_1));

        lane_markings_publisher_ = this->create_publisher<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>(
            "/lane_markings_in_odom", 10);

        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>(
            "/reference_path", 10);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        RCLCPP_INFO(this->get_logger(), "Lane Transformer Node Initialized.");
    }

private:
    void laneMarkingsCallback(const lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth::SharedPtr msg)
    {
        geometry_msgs::msg::TransformStamped transform;
        try
        {
            transform = tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Failed to get transform: %s", ex.what());
            return;
        }

        lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth transformed_msg;
        transformed_msg.header.stamp = this->get_clock()->now();
        transformed_msg.header.frame_id = "odom";

        nav_msgs::msg::Path path_msg;
        path_msg.header = transformed_msg.header;

        size_t left_size = msg->markings_left.size();
        size_t right_size = msg->markings_right.size();
        if (left_size != right_size)
        {
            RCLCPP_WARN(this->get_logger(), "Left and right lane markings have different sizes: %zu vs %zu", left_size, right_size);
        }

        size_t min_size = std::min(left_size, right_size);
        for (size_t i = 0; i < min_size; ++i)
        {
            auto left_transformed = transformPoint(msg->markings_left[i], transform);
            auto right_transformed = transformPoint(msg->markings_right[i], transform);

            geometry_msgs::msg::PoseStamped pose;
            pose.header = path_msg.header;
            pose.pose.position.x = (left_transformed.x + right_transformed.x) / 2.0;
            pose.pose.position.y = (left_transformed.y + right_transformed.y) / 2.0;
            pose.pose.position.z = (left_transformed.z + right_transformed.z) / 2.0;
            pose.pose.orientation.w = 1.0;
            path_msg.poses.push_back(pose);

            transformed_msg.markings_left.push_back(left_transformed);
            transformed_msg.markings_right.push_back(right_transformed);
        }

        lane_markings_publisher_->publish(transformed_msg);

        path_publisher_->publish(path_msg);
    }

    lane_parameter_msg::msg::LaneMarkingProjected transformPoint(
        const lane_parameter_msg::msg::LaneMarkingProjected &point,
        const geometry_msgs::msg::TransformStamped &transform)
    {
        double tx = transform.transform.translation.x;
        double ty = transform.transform.translation.y;
        double tz = transform.transform.translation.z;

        tf2::Quaternion q(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w);
        tf2::Matrix3x3 rotation_matrix(q);

        lane_parameter_msg::msg::LaneMarkingProjected transformed_point;
        transformed_point.x = rotation_matrix[0][0] * point.x + rotation_matrix[0][1] * point.y +
                              rotation_matrix[0][2] * point.z + tx;
        transformed_point.y = rotation_matrix[1][0] * point.x + rotation_matrix[1][1] * point.y +
                              rotation_matrix[1][2] * point.z + ty;
        transformed_point.z = rotation_matrix[2][0] * point.x + rotation_matrix[2][1] * point.y +
                              rotation_matrix[2][2] * point.z + tz;

        return transformed_point;
    }

    rclcpp::Subscription<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>::SharedPtr lane_markings_subscription_;
    rclcpp::Publisher<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>::SharedPtr lane_markings_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LaneTransformerNode>());
    rclcpp::shutdown();
    return 0;
}

// add smooth function
#include <rclcpp/rclcpp.hpp> 
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include "lane_parameter_msg/msg/lane_marking_projected_array_both.hpp"
#include "lane_parameter_msg/msg/lane_marking_projected.hpp"
#include <vector>
#include <cmath>

class LaneTransformerNode : public rclcpp::Node
{
public:
    LaneTransformerNode() : Node("lane_transformer_node")
    {
        lane_markings_subscription_ = this->create_subscription<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>(
            "/lane_markings_projected", 10,
            std::bind(&LaneTransformerNode::laneMarkingsCallback, this, std::placeholders::_1));

        lane_markings_publisher_ = this->create_publisher<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>(
            "/lane_markings_in_odom", 10);

        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>(
            "/reference_path", 10);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        RCLCPP_INFO(this->get_logger(), "Lane Transformer Node Initialized.");
    }

private:
    void laneMarkingsCallback(const lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth::SharedPtr msg)
    {
        geometry_msgs::msg::TransformStamped transform;
        try
        {
            transform = tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Failed to get transform: %s", ex.what());
            return;
        }

        lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth transformed_msg;
        transformed_msg.header.stamp = this->get_clock()->now();
        transformed_msg.header.frame_id = "odom";

        nav_msgs::msg::Path path_msg;
        path_msg.header = transformed_msg.header;

        size_t left_size = msg->markings_left.size();
        size_t right_size = msg->markings_right.size();
        if (left_size != right_size)
        {
            RCLCPP_WARN(this->get_logger(), "Left and right lane markings have different sizes: %zu vs %zu", left_size, right_size);
        }

        size_t min_size = std::min(left_size, right_size);
        for (size_t i = 0; i < min_size; ++i)
        {
            auto left_transformed = transformPoint(msg->markings_left[i], transform);
            auto right_transformed = transformPoint(msg->markings_right[i], transform);

            geometry_msgs::msg::PoseStamped pose;
            pose.header = path_msg.header;
            pose.pose.position.x = (left_transformed.x + right_transformed.x) / 2.0;
            pose.pose.position.y = (left_transformed.y + right_transformed.y) / 2.0;
            pose.pose.position.z = (left_transformed.z + right_transformed.z) / 2.0;
            pose.pose.orientation.w = 1.0;
            path_msg.poses.push_back(pose);

            transformed_msg.markings_left.push_back(left_transformed);
            transformed_msg.markings_right.push_back(right_transformed);
        }

        // 对路径进行平滑处理
        path_msg.poses = smoothPath(path_msg.poses, 5);

        // 发布转换后的消息
        lane_markings_publisher_->publish(transformed_msg);

        // 发布平滑后的路径
        path_publisher_->publish(path_msg);
    }

    std::vector<geometry_msgs::msg::PoseStamped> smoothPath(const std::vector<geometry_msgs::msg::PoseStamped> &raw_path, int window_size)
    {
        std::vector<geometry_msgs::msg::PoseStamped> smoothed_path;
        int n = raw_path.size();
        if (n == 0 || window_size < 1)
            return smoothed_path;

        for (int i = 0; i < n; ++i)
        {
            double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
            int count = 0;

            // 平均窗口内点的坐标
            for (int j = std::max(0, i - window_size); j <= std::min(n - 1, i + window_size); ++j)
            {
                sum_x += raw_path[j].pose.position.x;
                sum_y += raw_path[j].pose.position.y;
                sum_z += raw_path[j].pose.position.z;
                count++;
            }

            geometry_msgs::msg::PoseStamped smoothed_pose = raw_path[i];
            smoothed_pose.pose.position.x = sum_x / count;
            smoothed_pose.pose.position.y = sum_y / count;
            smoothed_pose.pose.position.z = sum_z / count;

            smoothed_path.push_back(smoothed_pose);
        }
        return smoothed_path;
    }

    lane_parameter_msg::msg::LaneMarkingProjected transformPoint(
        const lane_parameter_msg::msg::LaneMarkingProjected &point,
        const geometry_msgs::msg::TransformStamped &transform)
    {
        double tx = transform.transform.translation.x;
        double ty = transform.transform.translation.y;
        double tz = transform.transform.translation.z;

        tf2::Quaternion q(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w);
        tf2::Matrix3x3 rotation_matrix(q);

        lane_parameter_msg::msg::LaneMarkingProjected transformed_point;
        transformed_point.x = rotation_matrix[0][0] * point.x + rotation_matrix[0][1] * point.y +
                              rotation_matrix[0][2] * point.z + tx;
        transformed_point.y = rotation_matrix[1][0] * point.x + rotation_matrix[1][1] * point.y +
                              rotation_matrix[1][2] * point.z + ty;
        transformed_point.z = rotation_matrix[2][0] * point.x + rotation_matrix[2][1] * point.y +
                              rotation_matrix[2][2] * point.z + tz;

        return transformed_point;
    }

    rclcpp::Subscription<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>::SharedPtr lane_markings_subscription_;
    rclcpp::Publisher<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>::SharedPtr lane_markings_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LaneTransformerNode>());
    rclcpp::shutdown();
    return 0;
}


