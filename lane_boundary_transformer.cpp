#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <custom_msgs/msg/lane_markings_projected.hpp> 

class LaneBoundaryTransformer : public rclcpp::Node
{
public:
    LaneBoundaryTransformer() : Node("lane_boundary_transformer")
    {
        // 初始化 TF Buffer 和 TransformListener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 订阅车道线信息
        lane_markings_subscription_ = this->create_subscription<custom_msgs::msg::LaneMarkingsProjected>(
            "/lane_markings_projected", 10,
            std::bind(&LaneBoundaryTransformer::laneMarkingsCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Lane Boundary Transformer Node Initialized.");
    }

private:
    void laneMarkingsCallback(const custom_msgs::msg::LaneMarkingsProjected::SharedPtr msg)
    {
        // 转换左右边界点
        auto left_boundary_in_odom = transformBoundaryToOdom(msg->markings_left);
        auto right_boundary_in_odom = transformBoundaryToOdom(msg->markings_right);

        // 输出转换后的结果（可以改为发布或保存）
        RCLCPP_INFO(this->get_logger(), "Left Boundary in Odom:");
        for (const auto &point : left_boundary_in_odom)
        {
            RCLCPP_INFO(this->get_logger(), "x=%.2f, y=%.2f, z=%.2f", point.x, point.y, point.z);
        }

        RCLCPP_INFO(this->get_logger(), "Right Boundary in Odom:");
        for (const auto &point : right_boundary_in_odom)
        {
            RCLCPP_INFO(this->get_logger(), "x=%.2f, y=%.2f, z=%.2f", point.x, point.y, point.z);
        }
    }

    // 将一条边界线转换到 odom 坐标系
    std::vector<geometry_msgs::msg::Point> transformBoundaryToOdom(
        const std::vector<geometry_msgs::msg::Point> &boundary_in_baselink)
    {
        std::vector<geometry_msgs::msg::Point> boundary_in_odom;

        try
        {
            // 获取 odom -> base_link 的变换
            geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
                "odom", "base_link", tf2::TimePointZero);

            for (const auto &point_in_baselink : boundary_in_baselink)
            {
                geometry_msgs::msg::Point point_in_odom;
                tf2::doTransform(point_in_baselink, point_in_odom, transform);
                boundary_in_odom.push_back(point_in_odom);
            }
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Failed to transform boundary: %s", ex.what());
        }

        return boundary_in_odom;
    }

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<custom_msgs::msg::LaneMarkingsProjected>::SharedPtr lane_markings_subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LaneBoundaryTransformer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
