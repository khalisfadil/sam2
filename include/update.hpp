#pragma once
#include <open3d/Open3D.h>
#include <map.hpp>
#include <sophus/se3.hpp>

inline std::shared_ptr<open3d::geometry::PointCloud> createPointCloud(const Points3fArray& points) {
    // Create a shared pointer to a new PointCloud
    auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();
    
    // Transform points from NED [N, E, D] to Open3D [E, N, D]
    std::vector<Eigen::Vector3d> open3d_points(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        open3d_points[i] = (Eigen::Vector3f(points[i](1), points[i](0), points[i](2))).cast<double>(); // [E, N, D]
    }
    
    // Assign transformed points to the PointCloud
    point_cloud->points_.assign(open3d_points.begin(), open3d_points.end());
    
    // Assign red color (1.0, 0.0, 0.0) to all points
    point_cloud->colors_.resize(points.size(), Eigen::Vector3d(1.0, 0.0, 0.0));
    
    return point_cloud;
}

inline std::shared_ptr<open3d::geometry::TriangleMesh> createVehicleMesh(const Sophus::SE3d& T) {
    auto vehicle_mesh = std::make_shared<open3d::geometry::TriangleMesh>();

    // Define local vertices for the vehicle (in vehicle’s local frame)
    std::vector<Eigen::Vector3d> local_vertices = {
        {10.0, 0.0, 0.0},   // Front tip (10 meters long along local X)
        {-10.0, -5.0, 0.0}, // Rear left
        {-10.0, 5.0, 0.0}   // Rear right
    };

    // Extract translation and rotation from T
    Eigen::Vector3d ned_pos = T.translation(); // [N, E, D]
    Eigen::Matrix3d ned_rot = T.rotationMatrix();

    // Option 1: Euler-based approach (as requested)
    // Convert rotation matrix to Euler angles (Z-Y-X convention, NED frame)
    double yaw = std::atan2(ned_rot(1, 0), ned_rot(0, 0)); // Z-axis (yaw)
    double pitch = std::asin(-ned_rot(2, 0));              // Y-axis (pitch)
    double roll = std::atan2(ned_rot(2, 1), ned_rot(2, 2)); // X-axis (roll)

    // Map NED Euler angles to Open3D frame (X-right, Y-up, Z-forward)
    // NED (X-north, Y-east, Z-down) -> Open3D (X-east, Y-north, Z-up)
    // Yaw (Z) remains on Z, pitch (Y) maps to X, roll (X) maps to Y
    Eigen::AngleAxisd yaw_open3d(yaw, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitch_open3d(pitch, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd roll_open3d(roll, Eigen::Vector3d::UnitY());
    Eigen::Matrix3d R_open3d = (yaw_open3d * pitch_open3d * roll_open3d).toRotationMatrix();

    // Map NED position to Open3D coordinates: [N, E, D] -> [E, N, -D]
    Eigen::Vector3d open3d_pos(ned_pos(1), ned_pos(0), ned_pos(2));

    // Create transformation matrix
    Eigen::Matrix4d T_open3d = Eigen::Matrix4d::Identity();
    T_open3d.block<3, 3>(0, 0) = R_open3d;
    T_open3d.block<3, 1>(0, 3) = open3d_pos;

    // Transform local vertices to world coordinates
    std::vector<Eigen::Vector3d> world_vertices(local_vertices.size());
    for (size_t i = 0; i < local_vertices.size(); ++i) {
        world_vertices[i] = T_open3d.block<3, 3>(0, 0) * local_vertices[i] + T_open3d.block<3, 1>(0, 3);
    }

    // Assign vertices, triangles, and colors
    vehicle_mesh->vertices_ = world_vertices;
    vehicle_mesh->triangles_.push_back(Eigen::Vector3i(0, 1, 2)); // Single triangular face
    vehicle_mesh->vertex_colors_ = {
        {0.0, 1.0, 0.0}, // Green
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0}
    };

    return vehicle_mesh;
}