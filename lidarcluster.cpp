// lidar_cluster_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

class LidarClusterNode : public rclcpp::Node {
public:
  LidarClusterNode() : Node("lidar_cluster_node") {
    declare_parameter<double>("cluster_tolerance", 0.5);
    declare_parameter<int>("min_cluster_size", 30);
    declare_parameter<int>("max_cluster_size", 25000);

    get_parameter("cluster_tolerance", cluster_tolerance_);
    get_parameter("min_cluster_size", min_cluster_size_);
    get_parameter("max_cluster_size", max_cluster_size_);

    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/scala_decoder_sdk_points_2", 10,
      std::bind(&LidarClusterNode::cloud_callback, this, std::placeholders::_1));

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/lidar_clusters", 10);
  }

private:
  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud);

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.2f, 0.2f, 0.2f);
    vg.filter(*cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    visualization_msgs::msg::MarkerArray marker_array;
    int id = 0;
    rclcpp::Time now = this->now();
    for (const auto& indices : cluster_indices) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>());
      for (int idx : indices.indices)
        cluster->points.push_back(cloud->points[idx]);

      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cluster, centroid);
      Eigen::Vector4f min_pt, max_pt;
      pcl::getMinMax3D(*cluster, min_pt, max_pt);

      visualization_msgs::msg::Marker box;
      box.header = msg->header;
      box.ns = "clusters";
      box.id = id++;
      box.type = visualization_msgs::msg::Marker::CUBE;
      box.action = visualization_msgs::msg::Marker::ADD;
      box.pose.position.x = (min_pt.x() + max_pt.x()) / 2.0;
      box.pose.position.y = (min_pt.y() + max_pt.y()) / 2.0;
      box.pose.position.z = (min_pt.z() + max_pt.z()) / 2.0;
      box.scale.x = max_pt.x() - min_pt.x();
      box.scale.y = max_pt.y() - min_pt.y();
      box.scale.z = max_pt.z() - min_pt.z();
      box.color.r = 1.0;
      box.color.g = 0.0;
      box.color.b = 0.0;
      box.color.a = 0.5;
      box.lifetime = rclcpp::Duration::from_seconds(0.5);

      marker_array.markers.push_back(box);
    }

    marker_pub_->publish(marker_array);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarClusterNode>());
  rclcpp::shutdown();
  return 0;
} 
