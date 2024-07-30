#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <array>

#define PORT 5501
#define BUFFER_SIZE 48  // 6 doubles * 8 bytes per double

class UDPReceiver : public rclcpp::Node
{
public:
    UDPReceiver() : Node("udp_receiver")
    {
        publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("udp_data", 10);
        receive_thread_ = std::thread(&UDPReceiver::receiveData, this);
    }

    ~UDPReceiver()
    {
        running_ = false;
        if (receive_thread_.joinable())
        {
            receive_thread_.join();
        }
        close(sockfd_);
    }

private:
    void receiveData()
    {
        sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Socket creation failed");
            return;
        }

        struct sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(PORT);
        server_addr.sin_addr.s_addr = inet_addr("10.42.0.12");

        if (bind(sockfd_, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Bind failed");
            close(sockfd_);
            return;
        }

        while (running_)
        {
            char buffer[BUFFER_SIZE];
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            int n = recvfrom(sockfd_, buffer, BUFFER_SIZE, 0, (struct sockaddr *)&client_addr, &addr_len);
            if (n > 0)
            {
                std_msgs::msg::Float64MultiArray msg;
                double *data = reinterpret_cast<double*>(buffer);
                msg.data = std::vector<double>(data, data + 6);
                publisher_->publish(msg);
            }
        }
    }

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    std::thread receive_thread_;
    int sockfd_;
    bool running_ = true;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UDPReceiver>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}



下面是创建一个 ROS 2 Foxy 包的完整步骤，包括自定义消息和节点，以接收 `udp_receive_data` 主题的消息并将其转换为自定义的 `vehicle_dynamic` 消息。

### 创建 ROS 2 包

首先，创建一个新的 ROS 2 包：

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake vehicle_dynamic_pkg --dependencies rclcpp std_msgs
```

### 创建自定义消息

在 `vehicle_dynamic_pkg/msg` 目录下创建一个新的消息文件 `VehicleDynamic.msg`，内容如下：

```plaintext
float32 steering_angle
float32 accelerate_y
float32 yaw_rate
float32 accelerate_x
float32 yaw_accelerate
float32 car_speed
float32 wheel_speed_fl
float32 wheel_speed_fr
float32 wheel_speed_rl
float32 wheel_speed_rr
```

### 编辑 `CMakeLists.txt` 文件

编辑 `CMakeLists.txt` 文件，确保包含自定义消息的生成：

```cmake
cmake_minimum_required(VERSION 3.5)
project(vehicle_dynamic_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VehicleDynamic.msg"
)

ament_export_dependencies(rosidl_default_runtime)

add_executable(vehicle_dynamic_node src/vehicle_dynamic_node.cpp)
ament_target_dependencies(vehicle_dynamic_node rclcpp std_msgs)

install(TARGETS
  vehicle_dynamic_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
```

### 编辑 `package.xml` 文件

确保 `package.xml` 文件包含以下内容：

```xml
<package format="2">
  <name>vehicle_dynamic_pkg</name>
  <version>0.0.0</version>
  <description>Vehicle Dynamic Package</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <build_depend>rclcpp</build_depend>
  <build_depend>std_msgs</build_depend>
  <build_depend>rosidl_default_generators</build_depend>

  <exec_depend>rclcpp</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>

  <member_of_group>rosidl_interface_packages</member_of_group>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### 编写节点

在 `vehicle_dynamic_pkg/src` 目录下创建一个新的 C++ 文件 `vehicle_dynamic_node.cpp`，内容如下：

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include "vehicle_dynamic_pkg/msg/vehicle_dynamic.hpp"

using std::placeholders::_1;

class VehicleDynamicNode : public rclcpp::Node
{
public:
    VehicleDynamicNode() : Node("vehicle_dynamic_node")
    {
        subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "udp_receive_data", 10, std::bind(&VehicleDynamicNode::topic_callback, this, _1));
        publisher_ = this->create_publisher<vehicle_dynamic_pkg::msg::VehicleDynamic>("vehicle_dynamic_data", 10);
    }

private:
    void topic_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        if (msg->data.size() == 10)
        {
            auto vehicle_msg = vehicle_dynamic_pkg::msg::VehicleDynamic();
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
    rclcpp::Publisher<vehicle_dynamic_pkg::msg::VehicleDynamic>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VehicleDynamicNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
```

### 编译包

在工作空间根目录下编译包：

```bash
cd ~/ros2_ws
colcon build --packages-select vehicle_dynamic_pkg
source install/setup.bash
```

### 运行节点

运行节点：

```bash
ros2 run vehicle_dynamic_pkg vehicle_dynamic_node
```

### 发布测试数据

你可以使用以下 Python 脚本发布 `udp_receive_data` 主题的测试数据：

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'udp_receive_data', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = Float32MultiArray()
        msg.data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TestPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

运行这个脚本发布测试数据：

```bash
python3 test_publisher.py
```

通过这些步骤，你就可以在 ROS 2 Foxy 中创建一个节点，接收 `udp_receive_data` 主题的 `Float32MultiArray` 消息，并将数据转换为自定义的 `vehicle_dynamic` 消息。

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import tf_transformations

class ImuTransform(Node):
    def __init__(self):
        super().__init__('imu_transform')
        self.subscription = self.create_subscription(
            Imu,
            '/camera/imu',
            self.imu_callback,
            10)
        self.publisher = self.create_publisher(Imu, '/imu_enu', 10)

    def imu_callback(self, msg):
        transformed_imu = Imu()
        transformed_imu.header = msg.header

        # Transform linear acceleration
        transformed_imu.linear_acceleration.x = msg.linear_acceleration.y
        transformed_imu.linear_acceleration.y = msg.linear_acceleration.x
        transformed_imu.linear_acceleration.z = -msg.linear_acceleration.z

        # Transform angular velocity
        transformed_imu.angular_velocity.x = msg.angular_velocity.y
        transformed_imu.angular_velocity.y = msg.angular_velocity.x
        transformed_imu.angular_velocity.z = -msg.angular_velocity.z

        # Transform orientation (assuming the orientation is quaternion)
        q = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )
        # Rotate 90 degrees around Z axis
        rotation = tf_transformations.quaternion_from_euler(0, 0, -1.5708)
        q_transformed = tf_transformations.quaternion_multiply(rotation, q)
        transformed_imu.orientation.x = q_transformed[0]
        transformed_imu.orientation.y = q_transformed[1]
        transformed_imu.orientation.z = q_transformed[2]
        transformed_imu.orientation.w = q_transformed[3]

        self.publisher.publish(transformed_imu)

def main(args=None):
    rclpy.init(args=args)
    imu_transform = ImuTransform()
    rclpy.spin(imu_transform)
    imu_transform.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
