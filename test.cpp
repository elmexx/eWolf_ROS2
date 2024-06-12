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
