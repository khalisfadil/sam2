#pragma once

#include <Eigen/Dense>
#include <robin_map.h>
#include <dataframe.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// %            ... type alias for 3D points array
using Points3fArray = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;

// %            ... defines a unique 3D voxel index using integer coordinate.
struct Voxel {
    int32_t x = 0; // voxel index in x-direction
    int32_t y = 0; // voxel index in y-direction
    int32_t z = 0; // voxel index in z-direction

    // %            ... voxel constructor
    Voxel(int x, int y, int z) : x(static_cast<int32_t>(x)), y(static_cast<int32_t>(y)), z(static_cast<int32_t>(z)) {}

    // %            ... equality operation for comparing 2 voxels
    bool operator==(const Voxel& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    // %            ... less than operation for comparing 2 voxels
    bool operator<(const Voxel& other) const {
        return x < other.x || (x == other.x && y < other.y) || (x == other.x && y == other.y && z < other.z);
    }

    // %            ... convert a point into voxel coordinates
    static Voxel coordinate(const Eigen::Vector3f& point, double voxel_size) {
        return {
            static_cast<int32_t>(point.x() / voxel_size),
            static_cast<int32_t>(point.y() / voxel_size),
            static_cast<int32_t>(point.z() / voxel_size)
        };
    }
};

// %            ... defines a voxel block for storing points
struct VoxelBlock {
    // %            ... constructor with capacity
    explicit VoxelBlock(int capacity = 20) : capacity(static_cast<int32_t>(capacity)) {
        points.reserve(capacity);
    }

    // %            ... checks if the block has reached its point capacity
    bool isfull() const {
        return points.size() >= static_cast<size_t>(capacity);
    }

    // %            ... returns the current number of points in the block
    int numpoints() const {
        return static_cast<int>(points.size());
    }

    // %            ... adds a point to the block if not full
    bool addpoint(const Eigen::Vector3f& point) {
        if (isfull()) return false;
        points.push_back(point);
        return true;
    }

    Points3fArray points; // list of 3D points contained within voxel
    int32_t lifetime = 0; // counter for expiration logic (0 means it never expires)
    int32_t capacity = 20; // maximum capacity in single block
};

// %            ... defines a custom hash function for the Voxel struct
struct VoxelHash {
    // %            ... hash combining pattern
    static size_t hash(const Voxel& voxel) {
        size_t seed = 0;
        seed ^= std::hash<int32_t>()(voxel.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int32_t>()(voxel.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int32_t>()(voxel.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    // %            ... hash equal comparison
    static bool equal(const Voxel& v1, const Voxel& v2) {
        return v1 == v2;
    }

    // %            ... operation for calling the static hash function
    size_t operator()(const Voxel& voxel) const {
        return hash(voxel);
    }
};

// %            ... type alias for a hash map
using VoxelHashMap = tsl::robin_map<Voxel, VoxelBlock, VoxelHash>;

// %            ... map class
class map {
public:
    // %            ... map constructor
    explicit map(int maplifetime) : lifetime(static_cast<int32_t>(maplifetime)) {}

    // %            ... returns the total number of points in the map
    [[nodiscard]] size_t size() const {
        size_t map_size = 0;
        for (auto& voxel : map_) {
            map_size += (voxel.second).numpoints();
        }
        return map_size;
    }

    // %            ... extracts all points from the map into a single vector
    [[nodiscard]] Points3fArray pointcloud() const {
        Points3fArray pointcloud;
        pointcloud.reserve(size());
        for (const auto& voxel : map_) {
            const auto& points = voxel.second.points;
            for (const auto& point : points) {
                pointcloud.push_back(point);
            }
        }
        return pointcloud;
    }

    // %            ... convert to PCL point cloud
    [[nodiscard]] pcl::PointCloud<pcl::PointXYZ>::Ptr toPCLPointCloud() const {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        cloud->reserve(size()); // Pre-allocate for efficiency
        for (const auto& voxel : map_) {
            const auto& points = voxel.second.points;
            for (const auto& point : points) {
                pcl::PointXYZ pcl_point;
                pcl_point.x = point.x();
                pcl_point.y = point.y();
                pcl_point.z = point.z();
                cloud->push_back(pcl_point);
            }
        }
        return cloud;
    }

    // %            ... remove far or expired voxels
    void remove(const Eigen::Vector3f& origin, float distance) {
        const double sq_distance = distance * distance;
        for (auto it = map_.begin(); it != map_.end();) {
            auto& voxel_block = it.value();
            voxel_block.lifetime -= 1;
            const auto& pt = voxel_block.points.front();
            if ((pt - origin).squaredNorm() > sq_distance || voxel_block.lifetime <= 0) {
                it = map_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // %            ... add points from Points3fArray
    void add(const Points3fArray& points, double voxel_size, double min_distance_points, int voxel_capacity) {
        for (const auto& point : points) {
            add(point, voxel_size, min_distance_points, voxel_capacity);
        }
    }

    // %            ... add points from vector of Point3f
    void add(const std::vector<Point3f>& points, double voxel_size, double min_distance_points, int voxel_capacity) {
        for (const auto& point : points) {
            add(point.pointsMap, voxel_size, min_distance_points, voxel_capacity);
        }
    }

    // %            ... add a single point to the map
    void add(const Eigen::Vector3f& point, double voxel_size, double min_distance_points, int voxel_capacity) {
        Voxel key = Voxel::coordinate(point, voxel_size);
        auto search = map_.find(key);
        if (search != map_.end()) {
            auto& voxel_block = search.value();
            if (!voxel_block.isfull()) {
                float sq_dist_min_to_points = voxel_size * voxel_size;
                for (int i = 0; i < voxel_block.numpoints(); ++i) {
                    auto& _point = voxel_block.points[i];
                    float sq_dist = (_point - point).squaredNorm();
                    if (sq_dist < sq_dist_min_to_points) {
                        sq_dist_min_to_points = sq_dist;
                    }
                }
                if (sq_dist_min_to_points > (min_distance_points * min_distance_points)) {
                    voxel_block.addpoint(point);
                }
            }
            voxel_block.lifetime = lifetime;
        } else {
            VoxelBlock block(voxel_capacity);
            block.addpoint(point);
            block.lifetime = lifetime;
            map_[key] = std::move(block);
        }
    }

    // %            ... find closest neighbor
    std::tuple<Eigen::Vector3f, float> closestneighbor(const Eigen::Vector3f& query, double voxel_size) const {
        Voxel key = Voxel::coordinate(query, voxel_size);
        Eigen::Vector3f closest_neighbor = Eigen::Vector3f::Zero();
        float closest_distance_sq = std::numeric_limits<float>::max();
        for (const auto& voxel_shift : shift) {
            Voxel query_voxel{key.x + voxel_shift.x, key.y + voxel_shift.y, key.z + voxel_shift.z};
            auto search = map_.find(query_voxel);
            if (search != map_.end() && !search->second.points.empty()) {
                const auto& points = search->second.points;
                const auto& neighbor = *std::min_element(
                    points.cbegin(), points.cend(), [&](const auto& lhs, const auto& rhs) {
                        return (lhs - query).squaredNorm() < (rhs - query).squaredNorm();
                    });
                float distance_sq = (neighbor - query).squaredNorm();
                if (distance_sq < closest_distance_sq) {
                    closest_neighbor = neighbor;
                    closest_distance_sq = distance_sq;
                }
            }
        }
        return std::make_tuple(closest_neighbor, std::sqrt(closest_distance_sq));
    }

private:
    const std::array<Voxel, 27> shift{{
        Voxel{0, 0, 0}, Voxel{1, 0, 0}, Voxel{-1, 0, 0}, Voxel{0, 1, 0}, Voxel{0, -1, 0},
        Voxel{0, 0, 1}, Voxel{0, 0, -1}, Voxel{1, 1, 0}, Voxel{1, -1, 0}, Voxel{-1, 1, 0},
        Voxel{-1, -1, 0}, Voxel{1, 0, 1}, Voxel{1, 0, -1}, Voxel{-1, 0, 1}, Voxel{-1, 0, -1},
        Voxel{0, 1, 1}, Voxel{0, 1, -1}, Voxel{0, -1, 1}, Voxel{0, -1, -1}, Voxel{1, 1, 1},
        Voxel{1, 1, -1}, Voxel{1, -1, 1}, Voxel{1, -1, -1}, Voxel{-1, 1, 1}, Voxel{-1, 1, -1},
        Voxel{-1, -1, 1}, Voxel{-1, -1, -1}}};

    VoxelHashMap map_;
    int32_t lifetime = 10;
};