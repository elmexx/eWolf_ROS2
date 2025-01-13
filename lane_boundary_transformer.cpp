#include <rclcpp/rclcpp.hpp>
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
        // 初始化订阅和发布
        lane_markings_subscription_ = this->create_subscription<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>(
            "/lane_markings_projected", 10,
            std::bind(&LaneTransformerNode::laneMarkingsCallback, this, std::placeholders::_1));

        lane_markings_publisher_ = this->create_publisher<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>(
            "/lane_markings_in_odom", 10);

        // 初始化 TF Buffer 和 Listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        RCLCPP_INFO(this->get_logger(), "Lane Transformer Node Initialized.");
    }

private:
    void laneMarkingsCallback(const lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth::SharedPtr msg)
    {
        // 获取变换 odom -> base_link
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

        // 创建新的车道线消息
        lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth transformed_msg;
        transformed_msg.header.stamp = this->get_clock()->now();
        transformed_msg.header.frame_id = "odom";

        // 变换左边车道线
        for (const auto &point : msg->markings_left)
        {
            transformed_msg.markings_left.push_back(transformPoint(point, transform));
        }

        // 变换右边车道线
        for (const auto &point : msg->markings_right)
        {
            transformed_msg.markings_right.push_back(transformPoint(point, transform));
        }

        // 发布变换后的消息
        lane_markings_publisher_->publish(transformed_msg);
    }

    lane_parameter_msg::msg::LaneMarkingProjected transformPoint(
        const lane_parameter_msg::msg::LaneMarkingProjected &point,
        const geometry_msgs::msg::TransformStamped &transform)
    {
        // 提取平移和旋转信息
        double tx = transform.transform.translation.x;
        double ty = transform.transform.translation.y;
        double tz = transform.transform.translation.z;

        tf2::Quaternion q(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w);
        tf2::Matrix3x3 rotation_matrix(q);

        // 应用变换
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
