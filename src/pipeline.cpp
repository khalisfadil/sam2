#include <pipeline.hpp>

using json = nlohmann::json;

std::atomic<bool> pipeline::RUNNING_{true};
std::condition_variable pipeline::CV_;
std::atomic<bool> pipeline::MAPTOOGLE_{true};

// %            ... initgravity
pipeline::pipeline(const std::string& LIDARMETA, const std::string& LIDARCONFIG, 
                  const std::string& IMUCONFIG, const std::string& SYSCONFIG)
    : OPTIONS_(parseOption(SYSCONFIG)), 
      LIDARCALLBACK_(LIDARMETA, LIDARCONFIG), 
      NAVCALLBACK_(IMUCONFIG), 
      MAP_(OPTIONS_.maplifetime),
      MAPBUFFER1_(std::make_shared<Points3fArray>()),
      MAPBUFFER2_(std::make_shared<Points3fArray>()),
      TBUFFER1_(std::make_shared<Sophus::SE3d>()),
      TBUFFER2_(std::make_shared<Sophus::SE3d>()) {
        
    if(OPTIONS_.registration_method == "NDT_OMP"){
        pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp(new pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
        ndt_omp->setResolution(OPTIONS_.ndt_resolution);
        ndt_omp->setTransformationEpsilon(OPTIONS_.ndt_transform_epsilon);
        ndt_omp->setNumThreads(OPTIONS_.num_threads);
        if(OPTIONS_.ndt_neighborhood_search_method == "DIRECT1"){ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT1);
        }else if(OPTIONS_.ndt_neighborhood_search_method == "DIRECT7"){ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
        }else if(OPTIONS_.ndt_neighborhood_search_method == "KDTREE"){ndt_omp->setNeighborhoodSearchMethod(pclomp::KDTREE);
        }else{ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT1);}
        REGISTRATION_ = ndt_omp;
    }else if(OPTIONS_.registration_method == "GICP"){
        pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr gicp(new pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>());
        gicp->setMaxCorrespondenceDistance(OPTIONS_.gicp_corr_dist_threshold);
        gicp->setTransformationEpsilon(OPTIONS_.gicp_transform_epsilon);
        REGISTRATION_ = gicp;
    }else {
        throw std::runtime_error("Registration method is not valid.");
    }
#ifdef DEBUG
    logMessage("LOGGING", "pipeline initialized.");
#endif
}
// %            ... initgravity
Options pipeline::parseOption(const std::string& JSON) {
    std::ifstream file(JSON);
        if (!file.is_open()) {throw std::runtime_error("Failed to open JSON file: " + JSON);}
    nlohmann::json json_data;
    try {
        file >> json_data;
    } catch (const nlohmann::json::parse_error& e) {throw std::runtime_error("JSON parse error in " + JSON + ": " + e.what());}
    Options parsed_options;
    if (!json_data.is_object()) {throw std::runtime_error("JSON data must be an object");}
    try {
        if (!json_data.contains("slam_configuration") || !json_data["slam_configuration"].is_object()) {throw std::runtime_error("Missing or invalid 'odometry_options' object");}
        const auto& slam_configuration = json_data["slam_configuration"];

        if (slam_configuration.contains("intializestep")) parsed_options.intializestep = slam_configuration["intializestep"].get<int>();
        if (slam_configuration.contains("num_threads")) parsed_options.num_threads = slam_configuration["num_threads"].get<int>();

        if (slam_configuration.contains("maplifetime")) parsed_options.maplifetime = slam_configuration["maplifetime"].get<int>();
        if (slam_configuration.contains("mapvoxelsize")) parsed_options.mapvoxelsize = slam_configuration["mapvoxelsize"].get<float>();
        if (slam_configuration.contains("mapvoxelcapacity")) parsed_options.mapvoxelcapacity = slam_configuration["mapvoxelcapacity"].get<int>();
        if (slam_configuration.contains("mapmaxdistance")) parsed_options.mapmaxdistance = slam_configuration["mapmaxdistance"].get<float>();

        if (slam_configuration.contains("registration_method")) parsed_options.registration_method = slam_configuration["registration_method"].get<std::string>();
        if (slam_configuration.contains("ndt_resolution")) parsed_options.ndt_resolution = slam_configuration["ndt_resolution"].get<float>();
        if (slam_configuration.contains("ndt_transform_epsilon")) parsed_options.ndt_transform_epsilon = slam_configuration["ndt_transform_epsilon"].get<float>();
        if (slam_configuration.contains("ndt_neighborhood_search_method")) parsed_options.ndt_neighborhood_search_method = slam_configuration["ndt_neighborhood_search_method"].get<std::string>();
        
        if (slam_configuration.contains("gicp_corr_dist_threshold")) parsed_options.gicp_corr_dist_threshold = slam_configuration["gicp_corr_dist_threshold"].get<float>();
        if (slam_configuration.contains("gicp_transform_epsilon")) parsed_options.gicp_transform_epsilon = slam_configuration["gicp_transform_epsilon"].get<float>();

    } catch (const nlohmann::json::exception& e) {throw std::runtime_error("JSON parsing error in metadata: " + std::string(e.what()));}
    return parsed_options;
}
// %            ... initgravity
void pipeline::signalHandler(int SIGNAL) {
    if (SIGNAL == SIGINT || SIGNAL == SIGTERM) {
        RUNNING_.store(false, std::memory_order_release);
        CV_.notify_all();
    }
}
// %            ... initgravity
void pipeline::setThreadAffinity(const std::vector<int>& COREIDS) {
    if (COREIDS.empty()) {return;}
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    const unsigned int maxCores = std::thread::hardware_concurrency();
    uint32_t validCores = 0;
    for (int coreID : COREIDS) {
        if (coreID >= 0 && static_cast<unsigned>(coreID) < maxCores) {
            CPU_SET(coreID, &cpuset);
            validCores |= (1 << coreID);
        }
    }
    if (!validCores) {return;}
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        RUNNING_.store(false); // Optionally terminate
    }
}
// %            ... initgravity
void pipeline::logMessage(const std::string& LEVEL, const std::string& MESSAGE) {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << "[" << std::put_time(std::gmtime(&now_time_t), "%Y-%m-%dT%H:%M:%SZ") << "] "
        << "[" << LEVEL << "] " << MESSAGE << "\n";
    if (!LOGQUEUE_.push(oss.str())) {
        DROPLOG_.fetch_add(1, std::memory_order_relaxed);
    }
}
// %            ... initgravity
void pipeline::processLogQueue(const std::string& FILENAME, const std::vector<int>& ALLOWCORES) {
    setThreadAffinity(ALLOWCORES); // Pin logging thread to specified cores

    std::ofstream outfile(FILENAME);
    if (!outfile.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << "[" << std::put_time(std::gmtime(&now_time_t), "%Y-%m-%dT%H:%M:%SZ") << "] "
            << "[ERROR] failed to open file " << FILENAME << " for writing.\n";
        std::cerr << oss.str(); // Fallback to cerr if file cannot be opened
        return;
    }

    std::string message;
    while (RUNNING_.load(std::memory_order_acquire)) {
        if (LOGQUEUE_.pop(message)) {
            outfile << message;
            int currentDrops = DROPLOG_.load(std::memory_order_relaxed);
            if (currentDrops > LASTDROP_ && (currentDrops - LASTDROP_) >= 100) {
                auto now = std::chrono::system_clock::now();
                auto now_time_t = std::chrono::system_clock::to_time_t(now);
                std::ostringstream oss;
                oss << "[" << std::put_time(std::gmtime(&now_time_t), "%Y-%m-%dT%H:%M:%SZ") << "] "
                    << "[WARNING] " << (currentDrops - LASTDROP_) << " log messages dropped due to queue overflow.\n";
                outfile << oss.str();
                LASTDROP_ = currentDrops;
            }
        } else {
            std::this_thread::yield();
        }
    }
    while (LOGQUEUE_.pop(message)) {
        outfile << message;
    }
    int finalDrops = DROPLOG_.load(std::memory_order_relaxed);
    if (finalDrops > LASTDROP_) {

        outfile << "[LOGGING] Final report: " << (finalDrops - LASTDROP_) << " log messages dropped.\n";
    }

    outfile.flush(); // Ensure data is written
    outfile.close();
}
// %            ... initgravity
void pipeline::runLidarListenerRng19(boost::asio::io_context& IOCONTEXT, UdpSocketConfig UDPCONFIG, const std::vector<int>& ALLOWCORES) {
    setThreadAffinity(ALLOWCORES);
    RUNNING_.store(true, std::memory_order_release);
    // Validate UdpSocketConfig
    if (UDPCONFIG.host.empty() || UDPCONFIG.port == 0 || UDPCONFIG.bufferSize == 0) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Lidar Listener: Invalid host, port, or buffer size. Host: " << UDPCONFIG.host 
            << ", Port: " << UDPCONFIG.port << ", Buffer: " << UDPCONFIG.bufferSize;
        logMessage("ERROR", oss.str());
#endif
        return;
    }
    if (UDPCONFIG.receiveTimeout && UDPCONFIG.receiveTimeout->count() <= 0) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Lidar Listener: Invalid receive timeout: " << UDPCONFIG.receiveTimeout->count() << " ms";
        logMessage("ERROR", oss.str());
#endif
        return;
    }
    if (UDPCONFIG.multicastGroup && !UDPCONFIG.multicastGroup->is_multicast()) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Lidar Listener: Invalid multicast group: " << UDPCONFIG.multicastGroup->to_string();
        logMessage("ERROR", oss.str());
#endif
        return;
    }
    try {
        // Data callback: Decode and push to SPSC buffer
        // auto dataCallback = [this](SpanType packet_data) {
        auto dataCallback = [this](DataBuffer packet_data) {
            
            try {
                LidarFrame lidarframe;
                LIDARCALLBACK_.DecodePacketRng19(
                    std::vector<uint8_t>(packet_data.begin(), packet_data.end()), lidarframe);
                if (lidarframe.numberpoints > 0 && lidarframe.frame_id != LIDARFRAMEID_) {
                    LIDARFRAMEID_ = lidarframe.frame_id;
                    if (!BUFFLIDARFRAME_.push(std::move(lidarframe))) {
#ifdef DEBUG
                        logMessage("WARNING", "Lidar Listener: SPSC buffer push failed for frame " + 
                                   std::to_string(LIDARFRAMEID_));
#endif
                    } else {
#ifdef DEBUG
                        std::ostringstream oss;
                        oss << "Lidar Listener: Processed frame " << LIDARFRAMEID_ << " with " 
                            << lidarframe.numberpoints << " points";
                        logMessage("LOGGING", oss.str());
#endif
                    }
                }
            } catch (const std::exception& e) {
#ifdef DEBUG
                std::ostringstream oss;
                oss << "Lidar Listener: Decode error: " << e.what();
                logMessage("WARNING", oss.str());
#endif
            }
        };
        // Error callback: Log errors and handle shutdown
        auto errorCallback = [this](const boost::system::error_code& ec) {
#ifdef DEBUG
            std::ostringstream oss;
            oss << "Lidar Listener: UDP error: " << ec.message() << " (value: " << ec.value() << ")";
            logMessage("WARNING", oss.str());
#endif
            if (ec == boost::asio::error::operation_aborted || !RUNNING_.load(std::memory_order_acquire)) {
                RUNNING_.store(false, std::memory_order_release);
            }
        };
        // Create UDP socket
        auto socket = UdpSocket::create(IOCONTEXT, UDPCONFIG, dataCallback, errorCallback);
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Lidar Listener: Started on " << UDPCONFIG.host << ":" << UDPCONFIG.port 
            << " with buffer size " << UDPCONFIG.bufferSize;
        logMessage("LOGGING", oss.str());
#endif
        // Run io_context (single-threaded, non-blocking)
        while (RUNNING_.load(std::memory_order_acquire)) {
            try {
                IOCONTEXT.run_one(); // Process one event at a time
            } catch (const std::exception& e) {
#ifdef DEBUG
                std::ostringstream oss;
                oss << "Lidar Listener: io_context exception: " << e.what();
                logMessage("WARNING", oss.str());
#endif
                if (RUNNING_.load(std::memory_order_acquire)) {
                    IOCONTEXT.restart();
#ifdef DEBUG
                    logMessage("LOGGING", "Lidar Listener: io_context restarted.");
#endif
                } else {
                    break;
                }
            }
        }
        // Cleanup
        socket->stop();
        if (!IOCONTEXT.stopped()) {
            IOCONTEXT.stop();
#ifdef DEBUG
            logMessage("LOGGING", "Lidar Listener: Stopped");
#endif
        }

    } catch (const std::exception& e) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Lidar Listener: Setup exception: " << e.what();
        logMessage("WARNING", oss.str());
#endif
        RUNNING_.store(false, std::memory_order_release);
        if (!IOCONTEXT.stopped()) {
            IOCONTEXT.stop();
#ifdef DEBUG
            logMessage("LOGGING", "Lidar Listener: Stopped");
#endif
        }
    }
}
// %            ... initgravity
void pipeline::runLidarListenerLegacy(boost::asio::io_context& IOCONTEXT, UdpSocketConfig UDPCONFIG, const std::vector<int>& ALLOWCORES) {
    setThreadAffinity(ALLOWCORES);
    RUNNING_.store(true, std::memory_order_release);
    // Validate UdpSocketConfig
    if (UDPCONFIG.host.empty() || UDPCONFIG.port == 0 || UDPCONFIG.bufferSize == 0) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Lidar Listener: Invalid host, port, or buffer size. Host: " << UDPCONFIG.host 
            << ", Port: " << UDPCONFIG.port << ", Buffer: " << UDPCONFIG.bufferSize;
        logMessage("ERROR", oss.str());
#endif
        return;
    }
    if (UDPCONFIG.receiveTimeout && UDPCONFIG.receiveTimeout->count() <= 0) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Lidar Listener: Invalid receive timeout: " << UDPCONFIG.receiveTimeout->count() << " ms";
        logMessage("ERROR", oss.str());
#endif
        return;
    }
    if (UDPCONFIG.multicastGroup && !UDPCONFIG.multicastGroup->is_multicast()) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Lidar Listener: Invalid multicast group: " << UDPCONFIG.multicastGroup->to_string();
        logMessage("ERROR", oss.str());
#endif
        return;
    }
    try {
        // Data callback: Decode and push to SPSC buffer
        // auto dataCallback = [this](SpanType packet_data) {
        auto dataCallback = [this](DataBuffer packet_data) {
            try {
                LidarFrame lidarframe;
                LIDARCALLBACK_.DecodePacketLegacy(
                    std::vector<uint8_t>(packet_data.begin(), packet_data.end()), lidarframe);
                if (lidarframe.numberpoints > 0 && lidarframe.frame_id != LIDARFRAMEID_) {
                    LIDARFRAMEID_ = lidarframe.frame_id;
                    if (!BUFFLIDARFRAME_.push(std::move(lidarframe))) {
#ifdef DEBUG
                        logMessage("WARNING", "Lidar Listener: SPSC buffer push failed for frame " + 
                                   std::to_string(LIDARFRAMEID_));
#endif
                    } else {
#ifdef DEBUG
                        std::ostringstream oss;
                        oss << "Lidar Listener: Processed frame " << LIDARFRAMEID_ << " with " 
                            << lidarframe.numberpoints << " points";
                        logMessage("LOGGING", oss.str());
#endif
                    }
                }
            } catch (const std::exception& e) {
#ifdef DEBUG
                std::ostringstream oss;
                oss << "Lidar Listener: Decode error: " << e.what();
                logMessage("WARNING", oss.str());
#endif
            }
        };
        // Error callback: Log errors and handle shutdown
        auto errorCallback = [this](const boost::system::error_code& ec) {
#ifdef DEBUG
            std::ostringstream oss;
            oss << "Lidar Listener: UDP error: " << ec.message() << " (value: " << ec.value() << ")";
            logMessage("WARNING", oss.str());
#endif
            if (ec == boost::asio::error::operation_aborted || !RUNNING_.load(std::memory_order_acquire)) {
                RUNNING_.store(false, std::memory_order_release);
            }
        };
        // Create UDP socket
        auto socket = UdpSocket::create(IOCONTEXT, UDPCONFIG, dataCallback, errorCallback);
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Lidar Listener: Started on " << UDPCONFIG.host << ":" << UDPCONFIG.port 
            << " with buffer size " << UDPCONFIG.bufferSize;
        logMessage("LOGGING", oss.str());
#endif
        // Run io_context (single-threaded, non-blocking)
        while (RUNNING_.load(std::memory_order_acquire)) {
            try {
                IOCONTEXT.run_one(); // Process one event at a time
            } catch (const std::exception& e) {
#ifdef DEBUG
                std::ostringstream oss;
                oss << "Lidar Listener: io_context exception: " << e.what();
                logMessage("WARNING", oss.str());
#endif
                if (RUNNING_.load(std::memory_order_acquire)) {
                    IOCONTEXT.restart();
#ifdef DEBUG
                    logMessage("LOGGING", "Lidar Listener: io_context restarted.");
#endif
                } else {
                    break;
                }
            }
        }
        // Cleanup
        socket->stop();
        if (!IOCONTEXT.stopped()) {
            IOCONTEXT.stop();
#ifdef DEBUG
            logMessage("LOGGING", "Lidar Listener: Stopped");
#endif
        }

    } catch (const std::exception& e) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Lidar Listener: Setup exception: " << e.what();
        logMessage("WARNING", oss.str());
#endif
        RUNNING_.store(false, std::memory_order_release);
        if (!IOCONTEXT.stopped()) {
            IOCONTEXT.stop();
#ifdef DEBUG
            logMessage("LOGGING", "Lidar Listener: Stopped");
#endif
        }
    }
}
// %            ... initgravity
void pipeline::runNavListener(boost::asio::io_context& IOCONTEXT, UdpSocketConfig UDPCONFIG, const std::vector<int>& ALLOWCORES){
    setThreadAffinity(ALLOWCORES);
    RUNNING_.store(true, std::memory_order_release);
    // Validate UdpSocketConfig
    if (UDPCONFIG.host.empty() || UDPCONFIG.port == 0 || UDPCONFIG.bufferSize == 0) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Nav Listener: Invalid host, port, or buffer size. Host: " << UDPCONFIG.host
            << ", Port: " << UDPCONFIG.port << ", Buffer: " << UDPCONFIG.bufferSize;
        logMessage("ERROR", oss.str());
#endif
        return;
    }
    if (UDPCONFIG.receiveTimeout && UDPCONFIG.receiveTimeout->count() <= 0) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Nav Listener: Invalid receive timeout: " << UDPCONFIG.receiveTimeout->count() << " ms";
        logMessage("ERROR", oss.str());
#endif
        return;
    }
    if (UDPCONFIG.multicastGroup && !UDPCONFIG.multicastGroup->is_multicast()) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "Nav Listener: Invalid multicast group: " << UDPCONFIG.multicastGroup->to_string();
        logMessage("ERROR", oss.str());
#endif
        return;
    }

    try {
        // Data callback
        auto dataCallback = [this](DataBuffer packet_data) {
            try {
                CompFrame navframe;
                NAVCALLBACK_.Decode(std::vector<uint8_t>(packet_data.begin(), packet_data.end()), navframe);
                if (navframe.timestamp > 0 && navframe.timestamp != this->NAVTIMESTAMP_) {
                    this->NAVTIMESTAMP_ = navframe.timestamp;
                    // Maintain sliding window
                    if (NAVWINDOW_.size() >= NAVDATASIZE_) {
                        NAVWINDOW_.pop_front();
                    }
                    NAVWINDOW_.push_back(navframe);
                    if (NAVWINDOW_.size() == NAVDATASIZE_) {
                        if (!BUFFNAVWINFRAME_.push(NAVWINDOW_)) {
#ifdef DEBUG
                            logMessage("WARNING", "Nav Listener: Push to window frame failed for timestamp " +
                                       std::to_string(navframe.timestamp));
#endif
                        } else {
#ifdef DEBUG
                            std::ostringstream oss;
                            oss << "Nav Listener: Push to window frame for timestamp: " << navframe.timestamp
                                << ", Latitude: " << navframe.latitude << ", Longitude: " << navframe.longitude
                                << ", Altitude: " << navframe.altitude;
                            logMessage("LOGGING", oss.str());
#endif
                        }
                    }

                }
            } catch (const std::exception& e) {
#ifdef DEBUG
                std::ostringstream oss;
                oss << "runNavListener: Decode error: " << e.what();
                logMessage("WARNING", oss.str());
#endif
            }
        };

        // Error callback: Log errors and handle shutdown
        auto errorCallback = [this](const boost::system::error_code& ec) {
#ifdef DEBUG
            std::ostringstream oss;
            oss << "runNavListener: UDP error: " << ec.message() << " (value: " << ec.value() << ")";
            logMessage("WARNING", oss.str());
#endif
            if (ec == boost::asio::error::operation_aborted || !RUNNING_.load(std::memory_order_acquire)) {
                RUNNING_.store(false, std::memory_order_release);
            }
        };

        // Create UDP socket
        auto socket = UdpSocket::create(IOCONTEXT, UDPCONFIG, dataCallback, errorCallback);

        // Run io_context (single-threaded, non-blocking)
        while (RUNNING_.load(std::memory_order_acquire)) {
            try {
                IOCONTEXT.run_one(); // Process one event at a time
            } catch (const std::exception& e) {
#ifdef DEBUG
                std::ostringstream oss;
                oss << "runNavListener: io_context exception: " << e.what();
                logMessage("WARNING", oss.str());
#endif
                if (RUNNING_.load(std::memory_order_acquire)) {
                    IOCONTEXT.restart();
#ifdef DEBUG
                    logMessage("LOGGING", "runNavListener: io_context restarted.");
#endif
                }
            }
        }

        // Cleanup
        socket->stop();
        if (!IOCONTEXT.stopped()) {
            IOCONTEXT.stop();
#ifdef DEBUG
            logMessage("LOGGING", "runNavListener: Stopped");
#endif
        }

    } catch (const std::exception& e) {
#ifdef DEBUG
        std::ostringstream oss;
        oss << "runNavListener: Setup exception: " << e.what();
        logMessage("WARNING", oss.str());
#endif
        RUNNING_.store(false, std::memory_order_release);
        if (!IOCONTEXT.stopped()) {
            IOCONTEXT.stop();
#ifdef DEBUG
            logMessage("LOGGING", "runNavListener: Stopped");
#endif
        }
    }
}
// %            ... initgravity
void pipeline::dataAlignment(uint16_t STEPSIZE, const std::vector<int>& ALLOWCORES) {
    setThreadAffinity(ALLOWCORES);
    size_t alignment_count = 0; // Counter for successful alignments
    while (RUNNING_.load(std::memory_order_acquire)) {
        try {
            // 1. Pop a LiDAR frame from the buffer
            LidarFrame lidarframe;
            if (!BUFFLIDARFRAME_.pop(lidarframe) || lidarframe.timestamp_points.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            // 2. Validate and get the time range for the LiDAR scan
            if (lidarframe.timestamp_points.size() < 2) {
#ifdef DEBUG
                logMessage("ERROR", "dataAlignment: LidarDataFrame has insufficient points.");
#endif
                continue;
            }
            const double min_lidar_time = lidarframe.timestamp_points.front();
            const double max_lidar_time = lidarframe.timestamp_points.back();
            if (min_lidar_time > max_lidar_time) {
#ifdef DEBUG
                std::ostringstream oss;
                oss << "dataAlignment: Invalid LiDAR timestamp range (min: " << min_lidar_time
                    << ", max: " << max_lidar_time << ").";
                logMessage("ERROR", oss.str());
#endif
                continue;
            }
            // 3. Find a corresponding GNSS window
            bool aligned = false;
            while (!aligned && BUFFNAVWINFRAME_.read_available() > 0) {
                std::deque<CompFrame> navwinframe;
                if (!BUFFNAVWINFRAME_.pop(navwinframe) || navwinframe.empty()) {
                    break; // Try next LiDAR frame
                }
                if (navwinframe.size() < 2) {
#ifdef DEBUG
                    logMessage("WARNING", "dataAlignment: Nav packet has insufficient data points.");
#endif
                    continue;
                }
                const double min_nav_time = navwinframe.front().timestamp;
                const double max_nav_time = navwinframe.back().timestamp;
                if (min_nav_time > max_nav_time) {
#ifdef DEBUG
                    std::ostringstream oss;
                    oss << "dataAlignment: Invalid Nav timestamp range (min: " << min_nav_time
                        << ", max: " << max_nav_time << ").";
                    logMessage("WARNING", oss.str());
#endif
                    continue;
                }
                // Core Alignment Logic
                if (min_lidar_time >= min_nav_time && max_lidar_time <= max_nav_time) {
                    // Case 1: GNSS packet envelops LiDAR frame
                    aligned = true;
                    alignment_count++; // Increment alignment counter
#ifdef DEBUG
                    logMessage("LOGGING", "dataAlignment: Found alignment envelope.");
#endif
                    std::vector<CompFrame> filtnavwinframe;
                    if (!navwinframe.empty()) {
                        // Define interpolation function
                        auto getInterpolated = [&](double target_time) -> CompFrame {
                            if (target_time <= navwinframe.front().timestamp) {
                                return navwinframe.front();
                            }
                            if (target_time >= navwinframe.back().timestamp) {
                                return navwinframe.back();
                            }
                            for (size_t i = 0; i < navwinframe.size() - 1; ++i) {
                                const CompFrame& a = navwinframe[i];
                                const CompFrame& b = navwinframe[i + 1];
                                if (a.timestamp <= target_time && target_time <= b.timestamp) {
                                    double t = (target_time - a.timestamp) / (b.timestamp - a.timestamp);
                                    return navwinframe[i].linearInterpolate(a, b, t);
                                }
                            }
                            throw std::runtime_error("dataAlignment: Unable to interpolate at time " + std::to_string(target_time));
                        };

                        // Reserve space for efficiency
                        filtnavwinframe.reserve(navwinframe.size() + 2);
                        // Add interpolated start value at min_lidar_time
                        filtnavwinframe.push_back(getInterpolated(min_lidar_time));
                        // Add original values strictly between min_lidar_time and max_lidar_time
                        for (const auto& data : navwinframe) {
                            if (data.timestamp > min_lidar_time && data.timestamp < max_lidar_time) {
                                filtnavwinframe.push_back(data);
                            }
                        }
                        // Add interpolated end value at max_lidar_time
                        filtnavwinframe.push_back(getInterpolated(max_lidar_time));
                    }

                    if (!filtnavwinframe.empty()) {
                        FrameData frame;
                        frame.points = lidarframe.toPoint3f();
                        frame.timestamp = max_lidar_time;
                        // Pre-allocate or reserve space for imu and position vectors
                        frame.imu.reserve(filtnavwinframe.size()); // Reserve space for efficiency
                        frame.position.reserve(filtnavwinframe.size()); // Reserve space for efficiency
                        for (size_t i = 0; i < filtnavwinframe.size(); ++i) {
                            frame.imu.push_back(filtnavwinframe[i].toImuData());
                            frame.position.push_back(filtnavwinframe[i].toPositionData());
                        }
#ifdef DEBUG
                        std::ostringstream oss1, oss2;
                        oss1 << std::fixed << std::setprecision(12);
                        oss1 << "dataAlignment: LiDAR timestamp start: " << min_lidar_time
                             << ", timestamp end: " << lidarframe.timestamp_end;
                        logMessage("LOGGING", oss1.str());
                        oss2 << std::fixed << std::setprecision(12);
                        oss2 << "dataAlignment: Nav Window timestamp start: " << filtnavwinframe.front().timestamp
                             << ", timestamp end: " << filtnavwinframe.back().timestamp;
                        logMessage("LOGGING", oss2.str());
#endif
                        // Push to buffer only if alignment_count is a multiple of push_step_size
                        if (alignment_count % STEPSIZE == 0) {
                            if (!BUFFDATAFRAME_.push(std::move(frame))) {
#ifdef DEBUG
                                logMessage("WARNING", "dataAlignment: Failed to push Frame data to buffer.");
#endif
                            } else {
#ifdef DEBUG
                                std::ostringstream oss;
                                oss << "dataAlignment: Successfully pushed Frame data to buffer. Alignment count: " << alignment_count << " ###############!";
                                logMessage("LOGGING", oss.str());
#endif
                            }
                        } else {
#ifdef DEBUG
                            std::ostringstream oss;
                            oss << "dataAlignment: Skipped pushing combined data. Alignment count: " << alignment_count << " ###############!";
                            logMessage("LOGGING", oss.str());
#endif
                        }
                    } else {
#ifdef DEBUG
                        logMessage("WARNING", "dataAlignment: Alignment envelope found, but no GNSS points within LiDAR time span.");
#endif
                    }
                } else if (max_lidar_time > max_nav_time) {
                    // Case 2: LiDAR frame 
#ifdef DEBUG
                    logMessage("WARNING", "dataAlignment: LiDAR frame time exceed Navigation envelop. Discarding NAV envelop.");
#endif
                    continue;
                } else {
                    // Case 3: LiDAR frame is too old
#ifdef DEBUG
                    logMessage("WARNING", "dataAlignment: LiDAR frame time preceed Navigation envelop. Discarding LiDAR frame.");
#endif
                    break;
                }
            }
        } catch (const std::exception& e) {
#ifdef DEBUG
            std::ostringstream oss;
            oss << "dataAlignment: Exception occurred: " << e.what();
            logMessage("ERROR", oss.str());
#endif
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
#ifdef DEBUG
    logMessage("LOGGING", "dataAlignment: Stopped");
#endif
}
// %            ... initgravity
void pipeline::sam(const std::vector<int>& allowedCores) {
    setThreadAffinity(allowedCores);
    while (RUNNING_.load(std::memory_order_acquire)) {
        try {
            // Pop a Data frame from the buffer
            FrameData dataframe;
            if (!BUFFDATAFRAME_.pop(dataframe) || dataframe.timestamp < 0 || dataframe.points.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            const Eigen::Vector3d& euler = (dataframe.position.back().euler).cast<double>();
            const Eigen::Vector3d& pose = dataframe.position.back().pose;
            std::vector<Point3f> pointcloud = dataframe.points;
            Sophus::SE3d Tbc2m;
            if (INITIALIZESTEP_ >= 0) {
                // Initialization phase
                if (INITIALIZESTEP_ == OPTIONS_.intializestep) {
                    REFLLA_ = pose; // Set reference LLA
                }
                Eigen::Matrix3d Cb2m = (Eigen::AngleAxisd(euler.z(), Eigen::Vector3d::UnitZ()) *
                                       Eigen::AngleAxisd(euler.y(), Eigen::Vector3d::UnitY()) *
                                       Eigen::AngleAxisd(euler.x(), Eigen::Vector3d::UnitX())).matrix();
                Eigen::Vector3d tb2m = lla2ned(pose.x(), pose.y(), pose.z(), REFLLA_.x(), REFLLA_.y(), REFLLA_.z());
                Tbc2m = Sophus::SE3d(Cb2m, tb2m);
                Sophus::SE3d tbc2bp = TBP2M_.inverse() * Tbc2m;
                auto keypoints = initframe(pointcloud, tbc2bp);
                updatemap(keypoints, Tbc2m);
                INITIALIZESTEP_--;
                TBP2M_ = Tbc2m;
                TBC2BP_ = tbc2bp;
#ifdef DEBUG
                std::ostringstream oss;
                oss << "TBP2M_: Quaternion [x, y, z, w] = [" 
                    << TBP2M_.unit_quaternion().x() << ", " 
                    << TBP2M_.unit_quaternion().y() << ", " 
                    << TBP2M_.unit_quaternion().z() << ", " 
                    << TBP2M_.unit_quaternion().w() << "], "
                    << "Translation [x, y, z] = [" 
                    << TBP2M_.translation().x() << ", " 
                    << TBP2M_.translation().y() << ", " 
                    << TBP2M_.translation().z() << "]";
                logMessage("LOGGING", oss.str());
#endif
            } else {
                // Registration phase
                const Sophus::SE3d predTbc2m = TBP2M_ * TBC2BP_;
                auto keypoints = initframe(pointcloud, TBC2BP_);
                auto pc = toPCLPointCloud(keypoints);
                REGISTRATION_->setInputSource(pc);
                REGISTRATION_->setInputTarget(MAP_.toPCLPointCloud());
                pcl::PointCloud<pcl::PointXYZ>::Ptr pcout(new pcl::PointCloud<pcl::PointXYZ>);
                REGISTRATION_->align(*pcout, predTbc2m.matrix().cast<float>());
                if (!REGISTRATION_->hasConverged()) {
                    logMessage("ERROR", "sam: PCL registration failed to converge");
                    continue;
                }
                // Get and validate final transformation
                Eigen::Matrix4f T_float = REGISTRATION_->getFinalTransformation();
                Tbc2m = Sophus::SE3d(T_float.cast<double>());
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, keypoints.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t i = r.begin(); i != r.end(); ++i) {
                            auto& point = keypoints[i];
                            Eigen::Vector3d point_body_d = point.pointsBody.cast<double>();
                            Eigen::Vector3d point_map_d = Tbc2m * point_body_d;
                            point.pointsMap = point_map_d.cast<float>();
                        }
                    }
                );
                updatemap(keypoints, Tbc2m);
                TBC2BP_ = TBP2M_.inverse() * Tbc2m;
                TBP2M_ = Tbc2m;
#ifdef DEBUG
                std::ostringstream oss;
                oss << "TBP2M_: Quaternion [x, y, z, w] = [" 
                    << TBP2M_.unit_quaternion().x() << ", " 
                    << TBP2M_.unit_quaternion().y() << ", " 
                    << TBP2M_.unit_quaternion().z() << ", " 
                    << TBP2M_.unit_quaternion().w() << "], "
                    << "Translation [x, y, z] = [" 
                    << TBP2M_.translation().x() << ", " 
                    << TBP2M_.translation().y() << ", " 
                    << TBP2M_.translation().z() << "]";
                logMessage("LOGGING", oss.str());
#endif
            }
            // Update visualization buffers
            if (MAP_.pointcloud().empty()) {
                logMessage("ERROR", "sam: Empty point cloud in MAP_");
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                if (MAPTOOGLE_) {
                    *MAPBUFFER1_ = MAP_.pointcloud();
                    *TBUFFER1_ = Tbc2m;
                } else {
                    *MAPBUFFER2_ = MAP_.pointcloud();
                    *TBUFFER2_ = Tbc2m;
                }
                MAPTOOGLE_ = !MAPTOOGLE_;
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "sam: Exception occurred: " << e.what();
            logMessage("ERROR", oss.str());
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}
// %            ... inittimestamp
Eigen::Vector3d pipeline::lla2ned(double lat, double lon, double alt, double rlat, double rlon, double ralt) {
    // Constants according to WGS84
    constexpr double a = 6378137.0;              // Semi-major axis (m)
    constexpr double e2 = 0.00669437999014132;   // Squared eccentricity
    double dphi = lat - rlat;
    double dlam = SymmetricalAngle(lon - rlon);
    double dh = alt - ralt;
    double cp = std::cos(rlat);
    double sp = std::sin(rlat); // Fixed: was sin(originlon)
    double tmp1 = std::sqrt(1 - e2 * sp * sp);
    double tmp3 = tmp1 * tmp1 * tmp1;
    double dlam2 = dlam * dlam;   // Fixed: was dlam.*dlam
    double dphi2 = dphi * dphi;   // Fixed: was dphi.*dphi
    double E = (a / tmp1 + ralt) * cp * dlam -
            (a * (1 - e2) / tmp3 + ralt) * sp * dphi * dlam + // Fixed: was dphi.*dlam
            cp * dlam * dh;                                       // Fixed: was dlam.*dh
    double N = (a * (1 - e2) / tmp3 + ralt) * dphi +
            1.5 * cp * sp * a * e2 * dphi2 +
            sp * sp * dh * dphi +                              // Fixed: was dh.*dphi
            0.5 * sp * cp * (a / tmp1 + ralt) * dlam2;
    double D = -(dh - 0.5 * (a - 1.5 * a * e2 * cp * cp + 0.5 * a * e2 + ralt) * dphi2 -
                0.5 * cp * cp * (a / tmp1 - ralt) * dlam2);
    return Eigen::Vector3d(N, E, D);
}
// %            ... inittimestamp
Eigen::Vector3d pipeline::ned2lla(double n, double e, double d, double rlat, double rlon, double ralt) {
    // Constants and spheroid properties (WGS84)
    const double a = 6378137.0; // Semi-major axis (m)
    const double f = 1.0 / 298.257223563; // Flattening
    const double b = (1.0 - f) * a; // Semi-minor axis (m)
    const double e2 = f * (2.0 - f); // Square of first eccentricity
    const double ep2 = e2 / (1.0 - e2); // Square of second eccentricity
    double slat = std::sin(rlat);
    double clat = std::cos(rlat);
    double slon = std::sin(rlon);
    double clon = std::cos(rlon);
    double Nval = a / std::sqrt(1.0 - e2 * slat * slat);
    double rho = (Nval + ralt) * clat;
    double z0 = (Nval * (1.0 - e2) + ralt) * slat;
    double x0 = rho * clon;
    double y0 = rho * slon;
    double t = clat * (-d) - slat * n;
    double dz = slat * (-d) + clat * n;
    double dx = clon * t - slon * e;
    double dy = slon * t + clon * e;
    double x = x0 + dx;
    double y = y0 + dy;
    double z = z0 + dz;
    double lon = std::atan2(y, x);
    rho = std::hypot(x, y);
    double beta = std::atan2(z, (1.0 - f) * rho);
    double lat = std::atan2(z + b * ep2 * std::pow(std::sin(beta), 3),
                            rho - a * e2 * std::pow(std::cos(beta), 3));
    double betaNew = std::atan2((1.0 - f) * std::sin(lat), std::cos(lat));
    int count = 0;
    const int maxIterations = 5;
    while (std::abs(beta - betaNew) > 1e-10 && count < maxIterations) {
        beta = betaNew;
        lat = std::atan2(z + b * ep2 * std::pow(std::sin(beta), 3),
                         rho - a * e2 * std::pow(std::cos(beta), 3));
        betaNew = std::atan2((1.0 - f) * std::sin(lat), std::cos(lat));
        count++;
    }
    slat = std::sin(lat);
    Nval = a / std::sqrt(1.0 - e2 * slat * slat);
    double alt = rho * std::cos(lat) + (z + e2 * Nval * slat) * slat - Nval;
    return Eigen::Vector3d(lat, lon, alt);
}
// %            ... inittimestamp
double pipeline::SymmetricalAngle(double x) {
    constexpr double PI = M_PI;
    constexpr double TWO_PI = 2.0 * M_PI;
    double y = std::remainder(x, TWO_PI);
    if (y == PI) {y = -PI;}
    return y;
}
// %            ... inittimestamp
void pipeline::updatemap(std::vector<Point3f>& frame, const Sophus::SE3d& Tb2m) {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, frame.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                auto& point = frame[i];
                Eigen::Vector3d point_body_d = point.pointsBody.cast<double>();
                Eigen::Vector3d point_map_d = Tb2m * point_body_d;
                point.pointsMap = point_map_d.cast<float>();
            }
        }
    );
    const Eigen::Vector3d& origin = Tb2m.translation();
    double mindist = (OPTIONS_.mapvoxelsize * OPTIONS_.mapvoxelsize) / OPTIONS_.mapvoxelcapacity;
    MAP_.add(frame, OPTIONS_.mapvoxelsize, mindist, OPTIONS_.mapvoxelcapacity);
    MAP_.remove(origin.cast<float>(), OPTIONS_.mapmaxdistance);
}
// %            ... inittimestamp
std::vector<Point3f> pipeline::initframe(const std::vector<Point3f>& cframe, const Sophus::SE3d& Tbn2bp){
    const double vs = OPTIONS_.mapvoxelsize;
    std::vector<Point3f> keyframe;
    gridsampling(cframe, keyframe, vs);
    const auto &xi = Tbn2bp.log();
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, keyframe.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                auto& point = keyframe[i];
                float alpha = point.alpha;
                const auto pose = Sophus::SE3d::exp((static_cast<double>(alpha) - 1.0) * xi);
                Eigen::Vector3d point_body_d = point.pointsBody.cast<double>();
                point_body_d = pose * point_body_d;
                point.pointsBody = point_body_d.cast<float>();
            }
        }
    );
    return keyframe;
}
// %            ... inittimestamp
void pipeline::gridsampling(const std::vector<Point3f>& frame, std::vector<Point3f>& keypoints, double vs) {
    if (frame.empty()) {
        keypoints.clear();
        return;
    }
    using VoxelMap = tbb::concurrent_hash_map<Voxel, Point3f, VoxelHash>;
    VoxelMap vm;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, frame.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
                Voxel key = Voxel::coordinate(frame[i].pointsBody, vs);
                VoxelMap::accessor acc;
                vm.insert(acc, key);
                acc->second = frame[i];
            }
        }
    );
    keypoints.clear();
    keypoints.reserve(vm.size());
    for (const auto& pair : vm) {
        keypoints.push_back(pair.second);
    }
}
// %            ... inittimestamp
pcl::PointCloud<pcl::PointXYZ>::Ptr pipeline::toPCLPointCloud(std::vector<Point3f> points) const {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->reserve(points.size()); // Pre-allocate for efficiency
    for (const auto& point : points) {
        pcl::PointXYZ pcl_point;
        pcl_point.x = point.pointsBody.x();
        pcl_point.y = point.pointsBody.y();
        pcl_point.z = point.pointsBody.z();
        cloud->push_back(pcl_point);
    }
    return cloud;
}
// %            ... inittimestamp
void pipeline::runVizualization(const std::vector<int>& allowedCores) {
    setThreadAffinity(allowedCores);
    if (!VIS_.CreateVisualizerWindow("Map", 2560, 1440)) {
        logMessage("ERROR", "runVizualization: Failed to create visualizer window");
        return;
    }
    VIS_.GetRenderOption().background_color_ = Eigen::Vector3d(1, 1, 1); // White background
    // Initialize geometries
    if (!MAPPTR_) {
        MAPPTR_ = std::make_shared<open3d::geometry::PointCloud>();
    }
    if (!POSEPTR_) {
        POSEPTR_ = createVehicleMesh(std::make_shared<Sophus::SE3d>()); // Default at origin
    }
    // Add geometries
    VIS_.AddGeometry(MAPPTR_);
    VIS_.AddGeometry(POSEPTR_);

    // Add coordinate frame for reference
    auto coord_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(10.0);
    VIS_.AddGeometry(coord_frame);

    // Set initial view (top-down)
    auto& view = VIS_.GetViewControl();
    view.SetLookat({0.0, 0.0, 0.0});
    view.SetFront({0, 0, 1});
    view.SetUp({0, 1, 0});
    view.SetZoom(0.5);
    while (RUNNING_.load(std::memory_order_acquire)) {
        try {
            VIS_.RegisterAnimationCallback([&](open3d::visualization::Visualizer* vis_ptr) {
                try {
                    return updateVisualization(vis_ptr);
                } catch (const std::exception& e) {
                    logMessage("ERROR", "runVizualization: Animation callback exception: " + std::string(e.what()));
                    return false;
                }
            });
            VIS_.Run();
        } catch (const std::exception& e) {
            logMessage("ERROR", "runVizualization: Exception occurred: " + std::string(e.what()));
            RUNNING_.store(false, std::memory_order_release);
        }
    }
    VIS_.DestroyVisualizerWindow();
}
// %            ... inittimestamp
bool pipeline::updateVisualization(open3d::visualization::Visualizer* vis) {
    bool updated = false;
    constexpr double smoothingFactor = 0.1;
    Eigen::Vector3d ned_pos = Eigen::Vector3d::Zero();
    try {
        std::shared_ptr<open3d::geometry::PointCloud> new_map_ptr;
        std::shared_ptr<open3d::geometry::TriangleMesh> new_pose_ptr;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            if (MAPTOOGLE_) {
                new_map_ptr = createPointCloud(MAPBUFFER2_);
                new_pose_ptr = createVehicleMesh(TBUFFER2_);
                ned_pos = TBUFFER2_->translation();
            } else {
                new_map_ptr = createPointCloud(MAPBUFFER1_);
                new_pose_ptr = createVehicleMesh(TBUFFER1_);
                ned_pos = TBUFFER1_->translation();
            }
        }
        // Remove and re-add geometries
        vis->RemoveGeometry(MAPPTR_);
        vis->RemoveGeometry(POSEPTR_);
        MAPPTR_ = new_map_ptr;
        POSEPTR_ = new_pose_ptr;
        vis->AddGeometry(MAPPTR_);
        vis->AddGeometry(POSEPTR_);

        Eigen::Vector3d targetLookat = Eigen::Vector3d(ned_pos(1), ned_pos(0), ned_pos(2));
        CURRLOOK_ = CURRLOOK_ + smoothingFactor * (targetLookat - CURRLOOK_);
        auto& view = vis->GetViewControl();
        view.SetLookat({CURRLOOK_(0), CURRLOOK_(1), CURRLOOK_(2)});

        vis->PollEvents();
        vis->UpdateRender();
        updated = true;
    } catch (const std::exception& e) {
        logMessage("ERROR", "updateVisualization: Exception occurred: " + std::string(e.what()));
    }
    return updated;
}
// %            ... inittimestamp
std::shared_ptr<open3d::geometry::PointCloud> pipeline::createPointCloud(std::shared_ptr<Points3fArray> points) {
    if (!points || points->empty()) {
        logMessage("ERROR", "createPointCloud: Invalid or empty points array");
        return std::make_shared<open3d::geometry::PointCloud>();
    }
    auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud->points_.reserve(points->size());
    point_cloud->colors_.reserve(points->size());
    // Compute min/max z for dynamic coloring
    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < points->size(); ++i) {
        float z = (*points)[i](2);
        min_z = std::min(min_z, z);
        max_z = std::max(max_z, z);
    }
    float range = max_z - min_z + 1e-6; // Avoid division by zero
    for (size_t i = 0; i < points->size(); ++i) {
        point_cloud->points_.emplace_back((*points)[i](1), (*points)[i](0), (*points)[i](2)); // [E, N, D]
        float t = ((*points)[i](2) - min_z) / range; // Normalize z
        point_cloud->colors_.emplace_back(t, 0.0, 1.0 - t); // Blue to red gradient
    }
    return point_cloud;
}
// %            ... inittimestamp
std::shared_ptr<open3d::geometry::TriangleMesh> pipeline::createVehicleMesh(std::shared_ptr<Sophus::SE3d> T) {
    if (!T) {
        logMessage("ERROR", "createVehicleMesh: Invalid transformation pointer");
        return std::make_shared<open3d::geometry::TriangleMesh>();
    }
    auto vehicle_mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    // Define local vertices for a simple box vehicle (in local frame: length 10m, width 5m, height 2m)
    std::vector<Eigen::Vector3d> local_vertices = {
        {2.5, 5.0, 0.0}, {2.5, -5.0, 0.0}, {-2.5, -5.0, 0.0}, {-2.5, 5.0, 0.0}, // Bottom face
        {2.5, 5.0, 2.0}, {2.5, -5.0, 2.0}, {-2.5, -5.0, 2.0}, {-2.5, 5.0, 2.0}  // Top face
    };
    // Triangles for the box (12 triangles for 6 faces)
    std::vector<Eigen::Vector3i> triangles = {
        {0, 1, 2}, {0, 2, 3}, // Bottom
        {4, 5, 6}, {4, 6, 7}, // Top
        {0, 4, 5}, {0, 5, 1}, // Front
        {1, 5, 6}, {1, 6, 2}, // Right
        {2, 6, 7}, {2, 7, 3}, // Back
        {3, 7, 4}, {3, 4, 0}  // Left
    };
    // Extract translation and rotation from Sophus::SE3d
    Eigen::Vector3d ned_pos = T->translation(); // [N, E, D]
    Eigen::Matrix3d R_ned = T->so3().matrix();
    // Map NED rotation to Open3D coordinates (x=E, y=N, z=D)
    Eigen::Matrix3d R_open3d;
    R_open3d << R_ned(1, 1), R_ned(1, 0), R_ned(1, 2),
                R_ned(0, 1), R_ned(0, 0), R_ned(0, 2),
                R_ned(2, 1), R_ned(2, 0), R_ned(2, 2);
    // Create transformation matrix
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = R_open3d;
    transform.block<3, 1>(0, 3) = Eigen::Vector3d(ned_pos(1), ned_pos(0), ned_pos(2)); // [E, N, D]
    // Transform local vertices to world coordinates
    std::vector<Eigen::Vector3d> world_vertices(local_vertices.size());
    for (size_t i = 0; i < local_vertices.size(); ++i) {
        world_vertices[i] = transform.block<3, 3>(0, 0) * local_vertices[i] + transform.block<3, 1>(0, 3);
    }
    // Assign vertices, triangles, and colors
    vehicle_mesh->vertices_ = world_vertices;
    vehicle_mesh->triangles_ = triangles;
    vehicle_mesh->vertex_colors_.resize(local_vertices.size(), Eigen::Vector3d(0.0, 1.0, 0.0)); // Green
    return vehicle_mesh;
}





