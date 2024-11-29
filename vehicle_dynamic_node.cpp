#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include "vehicle_dynamic_msg/msg/vehicle_dynamic.hpp"

using std::placeholders::_1;

class VehicleDynamicNode : public rclcpp::Node
{
public:
    VehicleDynamicNode() : Node("vehicle_dynamic_node")
    {
        subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "udp_receive", 10, std::bind(&VehicleDynamicNode::topic_callback, this, _1));
        publisher_ = this->create_publisher<vehicle_dynamic_msg::msg::VehicleDynamic>("vehicle_dynamic_data", 10);
    }

private:
    void topic_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        if (msg->data.size() == 10)
        {
            auto vehicle_msg = vehicle_dynamic_msg::msg::VehicleDynamic();
            vehicle_msg.header.stamp = this->get_clock()->now();
            vehicle_msg.header.frame_id = "base_link";
            
            vehicle_msg.steering_angle = msg->data[0];
            vehicle_msg.accelerate_y = msg->data[1];
            vehicle_msg.yaw_rate = msg->data[2];
            vehicle_msg.accelerate_x = msg->data[3];
            vehicle_msg.yaw_accelerate = msg->data[4];
            vehicle_msg.car_speed = msg->data[5];
            vehicle_msg.wheel_speed_fl = msg->data[6];
            vehicle_msg.wheel_speed_fr = msg->data[7];
            vehicle_msg.wheel_speed_rl = msg->data[8];
            vehicle_msg.wheel_speed_rr = msg->data[9];

            publisher_->publish(vehicle_msg);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Received data size is not 10.");
        }
    }

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr subscription_;
    rclcpp::Publisher<vehicle_dynamic_msg::msg::VehicleDynamic>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VehicleDynamicNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
