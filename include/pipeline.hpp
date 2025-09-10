#pragma once

#include <boost/asio.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include <memory>
#include <thread>
#include <deque>

#include <map>
#include <udpsocket.hpp>
#include <dataframe.hpp>
#include <lidarcallback.hpp>
#include <compcallback.hpp>
#include <open3d/Open3D.h>
#include <update.hpp>

#include <pclomp/ndt_omp.h>
#include <pclomp/ndt_omp_impl.hpp>
#include <pclomp/voxel_grid_covariance_omp.h>
#include <pclomp/voxel_grid_covariance_omp_impl.hpp>
#include <pclomp/gicp_omp.h>
#include <pclomp/gicp_omp_impl.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>

#include <tbb/parallel_for.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>

class pipeline {
    public:
        static std::atomic<bool> RUNNING_;                                                          // 
        static std::condition_variable CV_;                                                          //
        std::atomic<int> DROPLOG_; 
        std::atomic<int> LASTDROP_;
                                                            //
        boost::lockfree::spsc_queue<std::string, boost::lockfree::capacity<16>> LOGQUEUE_;
        boost::lockfree::spsc_queue<LidarFrame, boost::lockfree::capacity<16>> BUFFLIDARFRAME_;         //
        boost::lockfree::spsc_queue<std::deque<CompFrame>, boost::lockfree::capacity<16>> BUFFNAVWINFRAME_;
        boost::lockfree::spsc_queue<FrameData, boost::lockfree::capacity<16>> BUFFDATAFRAME_;

        explicit pipeline(const std::string& LIDARMETA, const std::string& LIDARCONFIG, const std::string& IMUCONFIG, const std::string& SYSCONFIG);
        static void signalHandler(int SIGNAL);
        void setThreadAffinity(const std::vector<int>& COREIDS);
        void logMessage(const std::string& LEVEL, const std::string& MESSAGE);
        void processLogQueue(const std::string& FILENAME, const std::vector<int>& ALLOWCORES);
        void runLidarListenerRng19(boost::asio::io_context& IOCONTEXT, UdpSocketConfig UDPCONFIG, const std::vector<int>& ALLOWCORES);
        void runLidarListenerLegacy(boost::asio::io_context& IOCONTEXT, UdpSocketConfig UDPCONFIG, const std::vector<int>& ALLOWCORES); 
        void runNavListener(boost::asio::io_context& IOCONTEXT, UdpSocketConfig UDPCONFIG, const std::vector<int>& ALLOWCORES); 
        void dataAlignment(uint16_t STEPSIZE, const std::vector<int>& ALLOWCORES);
        void sam(const std::vector<int>& ALLOWCORES);
        void runVizualization(const std::vector<int>& allowedCores);

    private:
        Options OPTIONS_;
        // %              ... lidar callback
        LidarCallback LIDARCALLBACK_;
        uint16_t LIDARFRAMEID_ = 0;                                           // track frame id from lidar

        // %              ... navigation callback
        CompCallback NAVCALLBACK_;
        double NAVTIMESTAMP_ = 0.0;
        std::deque<CompFrame> NAVWINDOW_;
        const size_t NAVDATASIZE_ = 20;

        // %              ... sam
        int INITIALIZESTEP_ = 20;
        Eigen::Vector3d REFLLA_  = Eigen::Vector3d::Zero(); 
        Sophus::SE3d TBP2M_;     // always previous
        Sophus::SE3d TBC2BP_;
        map MAP_;
        pcl::Registration <pcl::PointXYZ, pcl::PointXYZ>::Ptr REGISTRATION_;
        static std::atomic<bool> MAPTOOGLE_; 
        std::shared_ptr<Points3fArray> MAPBUFFER1_;
        std::shared_ptr<Points3fArray> MAPBUFFER2_;
        std::shared_ptr<Sophus::SE3d> TBUFFER1_;
        std::shared_ptr<Sophus::SE3d> TBUFFER2_;

        // %              ... vizualization
        open3d::visualization::Visualizer VIS_;
        std::shared_ptr<open3d::geometry::PointCloud> MAPPTR_;                                     // Point cloud for map
        std::shared_ptr<open3d::geometry::TriangleMesh> POSEPTR_;
        Eigen::Vector3d CURRLOOK_;

        // %              ... Added for synchronization
        std::mutex buffer_mutex_;

        Options parseOption(const std::string& JSON);
        Eigen::Vector3d lla2ned(double lat, double lon, double alt, double rlat, double rlon, double ralt);
        Eigen::Vector3d ned2lla(double n, double e, double d, double rlat, double rlon, double ralt);
        double SymmetricalAngle(double x);
        void updatemap(std::vector<Point3f>& frame, const Sophus::SE3d& Tb2m);
        std::vector<Point3f> initframe(const std::vector<Point3f>& cframe, const Sophus::SE3d& Tbn2bp);
        void gridsampling(const std::vector<Point3f>& frame, std::vector<Point3f>& keypoints, double vs);
        pcl::PointCloud<pcl::PointXYZ>::Ptr toPCLPointCloud(std::vector<Point3f> points) const;
        bool updateVisualization(open3d::visualization::Visualizer* vis);
        std::shared_ptr<open3d::geometry::PointCloud> createPointCloud(std::shared_ptr<Points3fArray> points);
        std::shared_ptr<open3d::geometry::TriangleMesh> createVehicleMesh(std::shared_ptr<Sophus::SE3d> T);

};