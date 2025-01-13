#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <nav_msgs/msg/path.hpp>
#include <lane_parameter_msg/msg/lane_marking_projected_array_both.hpp>
#include <lane_parameter_msg/msg/lane_marking_projected.hpp>

class LaneBoundaryTransformer : public rclcpp::Node
{
public:
    LaneBoundaryTransformer() : Node("lane_boundary_transformer")
    {
    
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        lane_markings_subscription_ = this->create_subscription<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>(
            "/lane_markings_projected", 10,
            std::bind(&LaneBoundaryTransformer::laneMarkingsCallback, this, std::placeholders::_1));

        left_path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/lane_markings_left_path", 10);
        right_path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/lane_markings_right_path", 10);

        RCLCPP_INFO(this->get_logger(), "Lane Boundary Transformer Node Initialized.");
    }

private:
    void laneMarkingsCallback(const lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth::SharedPtr msg)
    {
        auto left_boundary_in_odom = transformBoundaryToOdom(msg->markings_left);
        auto right_boundary_in_odom = transformBoundaryToOdom(msg->markings_right);

        publishPath("/lane_markings_left_path", left_boundary_in_odom, msg->header.stamp);
        publishPath("/lane_markings_right_path", right_boundary_in_odom, msg->header.stamp);
    }

    std::vector<geometry_msgs::msg::PoseStamped> transformBoundaryToOdom(
        const std::vector<lane_parameter_msg::msg::LaneMarkingProjected> &boundary_in_baselink)
    {
        std::vector<geometry_msgs::msg::PoseStamped> boundary_in_odom;

        try
        {
            geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
                "odom", "base_link", tf2::TimePointZero);

            for (const auto &point_in_baselink : boundary_in_baselink)
            {
                geometry_msgs::msg::Point point_in_baselink_geom, point_in_odom_geom;
                point_in_baselink_geom.x = point_in_baselink.x;
                point_in_baselink_geom.y = point_in_baselink.y;
                point_in_baselink_geom.z = point_in_baselink.z;

                tf2::doTransform(point_in_baselink_geom, point_in_odom_geom, transform);

                geometry_msgs::msg::PoseStamped pose_stamped;
                pose_stamped.header.frame_id = "odom";
                pose_stamped.pose.position = point_in_odom_geom;
                pose_stamped.pose.orientation.w = 1.0; 

                boundary_in_odom.push_back(pose_stamped);
            }
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Failed to transform boundary: %s", ex.what());
        }

        return boundary_in_odom;
    }

    void publishPath(const std::string &topic, const std::vector<geometry_msgs::msg::PoseStamped> &poses, const rclcpp::Time &stamp)
    {
        auto path_msg = std::make_shared<nav_msgs::msg::Path>();
        path_msg->header.stamp = stamp;
        path_msg->header.frame_id = "odom";
        path_msg->poses = poses;

        if (topic == "/lane_markings_left_path")
        {
            left_path_publisher_->publish(*path_msg);
        }
        else if (topic == "/lane_markings_right_path")
        {
            right_path_publisher_->publish(*path_msg);
        }
    }

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Subscription<lane_parameter_msg::msg::LaneMarkingProjectedArrayBoth>::SharedPtr lane_markings_subscription_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr left_path_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr right_path_publisher_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LaneBoundaryTransformer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

// ros2 pkg create --build-type ament_cmake lane_boundary_transformer --dependencies rclcpp tf2_ros tf2_geometry_msgs geometry_msgs nav_msgs lane_parameter_msg

