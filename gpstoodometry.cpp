#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cmath>

class GpsToOdometry : public rclcpp::Node
{
public:
    GpsToOdometry()
        : Node("gps_to_odometry_node"),
          ref_lat_(0.0), ref_lon_(0.0), ref_set_(false)
    {
        gps_subscription_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/fix", 10, std::bind(&GpsToOdometry::gps_callback, this, std::placeholders::_1));
        odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry/gps", 10);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }

private:
    void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
        if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX)
        {
            RCLCPP_WARN(this->get_logger(), "No valid GPS fix.");
            return;
        }

        if (!ref_set_)
        {
            ref_lat_ = msg->latitude;
            ref_lon_ = msg->longitude;
            ref_alt_ = msg->altitude;
            ref_set_ = true;
            RCLCPP_INFO(this->get_logger(), "Reference point set: lat=%f, lon=%f, alt=%f", ref_lat_, ref_lon_, ref_alt_);
        }

        double x, y, z;
        gps_to_local(msg->latitude, msg->longitude, msg->altitude, x, y, z);

        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = msg->header.stamp;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";

        odom_msg.pose.pose.position.x = x;
        odom_msg.pose.pose.position.y = y;
        odom_msg.pose.pose.position.z = z;

        odom_msg.pose.pose.orientation.x = 0.0;
        odom_msg.pose.pose.orientation.y = 0.0;
        odom_msg.pose.pose.orientation.z = 0.0;
        odom_msg.pose.pose.orientation.w = 1.0;

        odometry_publisher_->publish(odom_msg);

        // TF (odom -> base_link)
        publish_dynamic_transform(msg->header.stamp, x, y, z);
    }

    void gps_to_local(double lat, double lon, double alt, double &x, double &y, double &z)
    {
        const double earth_radius = 6378137.0; 

        double lat_rad = lat * M_PI / 180.0;
        double lon_rad = lon * M_PI / 180.0;
        double ref_lat_rad = ref_lat_ * M_PI / 180.0;
        double ref_lon_rad = ref_lon_ * M_PI / 180.0;

        double delta_lat = lat_rad - ref_lat_rad;
        double delta_lon = lon_rad - ref_lon_rad;

        x = earth_radius * delta_lon * cos(ref_lat_rad); 
        y = earth_radius * delta_lat;                   
        z = alt - ref_alt_;                          
    }

    void publish_dynamic_transform(const rclcpp::Time &stamp, double x, double y, double z)
    {
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = stamp;
        transform.header.frame_id = "odom";
        transform.child_frame_id = "base_link";

        transform.transform.translation.x = x;
        transform.transform.translation.y = y;
        transform.transform.translation.z = z;

        transform.transform.rotation.x = 0.0;
        transform.transform.rotation.y = 0.0;
        transform.transform.rotation.z = 0.0;
        transform.transform.rotation.w = 1.0;

        tf_broadcaster_->sendTransform(transform);
    }

    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_subscription_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    double ref_lat_, ref_lon_, ref_alt_; 
    bool ref_set_;                       
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GpsToOdometry>());
    rclcpp::shutdown();
    return 0;
}
