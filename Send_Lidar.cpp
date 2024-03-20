#include <functional>
#include <stdlib.h>
#include <cstring>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <thread>
#include <ndi_sdk/Processing.NDI.Lib.h>


using std::placeholders::_1;

class NDI_Subscriber : public rclcpp::Node
{
public:
    NDI_Subscriber()
        : Node("ndi_stream"), p_frame_buffers{nullptr, nullptr}
    {       
        frame = 0;
        // Not required, but "correct" (see the SDK documentation).
        if (!NDIlib_initialize()) {
            // Cannot run NDI. Most likely because the CPU is not sufficient (see SDK documentation).
            // you can check this directly with a call to NDIlib_is_supported_CPU()
            RCLCPP_INFO(this->get_logger(), "Cannot run NDI.");
        }

        // Create an NDI source that is called "Valeo Lidar" and is clocked to the video.
        NDIlib_send_create_t NDI_send_create_desc;
        NDI_send_create_desc.p_ndi_name = "Valeo Lidar";

        pNDI_send_ = NDIlib_send_create(&NDI_send_create_desc);

        img_frame_ = this->create_subscription<sensor_msgs::msg::Image>(
            "lidar_topic", 10, std::bind(&NDI_Subscriber::topic_callback, this, _1));
    }

    ~NDI_Subscriber()
    {
        // Destroy the NDI sender
        NDIlib_send_destroy(pNDI_send_);

        // Free the allocated buffers
        free(p_frame_buffers[0]);
        free(p_frame_buffers[1]);

        // Not required, but nice
        NDIlib_destroy();
    }
    
    void addFrame()
    {
        frame++;
    }

private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) 
    {
        unsigned int pic_size = msg->width * msg->height * 3; // Assuming 3 bytes per pixel (RGB)

        // Allocate memory for frame buffers if not already done
        if (!p_frame_buffers[0]) {
            p_frame_buffers[0] = malloc(pic_size);
            p_frame_buffers[1] = malloc(pic_size);
        }

        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        uint8_t* picture = cv_ptr->image.data;

        // We are going to create a frame
        NDIlib_video_frame_v2_t NDI_video_frame;
        NDI_video_frame.xres = msg->width;
        NDI_video_frame.yres = msg->height;
        NDI_video_frame.FourCC = NDIlib_FourCC_type_RGBX;
        NDI_video_frame.line_stride_in_bytes = msg->width * 3; // Assuming 3 bytes per pixel (RGB)

        // Copy the image data to one of the frame buffers
        memcpy(p_frame_buffers[frame & 1], picture, pic_size);
        
        NDI_video_frame.p_data = (uint8_t*)p_frame_buffers[frame & 1];
        NDIlib_send_send_video_async_v2(pNDI_send_, &NDI_video_frame);
        
        RCLCPP_INFO(this->get_logger(), "Stream Image");
        addFrame();
    }
    
    unsigned int frame;
    void* p_frame_buffers[2];
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_frame_;
    NDIlib_send_instance_t pNDI_send_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NDI_Subscriber>());
    rclcpp::shutdown();
    return 0;
}
