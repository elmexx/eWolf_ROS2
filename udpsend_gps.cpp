#include <iostream>
#include <fstream>
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

int main() {
    std::string serial_port = "/dev/tty";
    std::ifstream gps_stream(serial_port);

    if (!gps_stream.is_open()) {
        std::cerr << "Failed to open serial port: " << serial_port << std::endl;
        return 1;
    }

    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    struct sockaddr_in servaddr;
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(5600);
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");  // IP

    std::string line;
    while (std::getline(gps_stream, line)) {
        if (!line.empty()) {
            ssize_t sent_size = sendto(sockfd, line.c_str(), line.size(), 0, (const struct sockaddr *)&servaddr, sizeof(servaddr));
            if (sent_size < 0) {
                std::cerr << "Failed to send data" << std::endl;
                break;
            } else {
                std::cout << "Sent: " << line << std::endl;
            }
        }
    }

    gps_stream.close();
    close(sockfd);
    return 0;
}
