#include <pipeline.hpp>
#include <udpsocket.hpp> // Make sure to include the new UdpSocket header
#include <nlohmann/json.hpp> // Assuming you have this for JSON parsing
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <csignal>

int main() {
    // Generate UTC timestamp for filename
    auto NOW = std::chrono::system_clock::now();
    auto UTCTIME = std::chrono::system_clock::to_time_t(NOW);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&UTCTIME), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    std::string IMUCONFIG = "../config/imu_config_berlin.json";
    std::string LIDARMETA = "../config/lidar_meta_berlin.json";
    std::string LIDARCONFIG = "../config/lidar_config_berlin.json";
    std::string SYSCONFIG = "../config/sys_config_berlin.json";
    uint32_t LIDARPACKETSIZE = 24896;
    std::string UDPPROFILELIDAR;
    uint16_t UDPPORTLIDAR;
    std::string UDPDESTINATION;
    uint16_t STEPSIZE = 1;
    pipeline pipeline(LIDARMETA, LIDARCONFIG, IMUCONFIG, SYSCONFIG);
    
    std::string log_filename = "../eval/log/log_report_" + timestamp + ".txt";

    try {
        std::ifstream json_file(LIDARMETA);
        if (!json_file.is_open()) {
            throw std::runtime_error("[Main] Error: Could not open JSON file: " + LIDARMETA);
        }
        nlohmann::json metadata_;
        json_file >> metadata_;
        json_file.close(); // Explicitly close the file

        if (!metadata_.contains("lidar_data_format") || !metadata_["lidar_data_format"].is_object()) {
            throw std::runtime_error("Missing or invalid 'lidar_data_format' object");
        }
        if (!metadata_.contains("config_params") || !metadata_["config_params"].is_object()) {
            throw std::runtime_error("Missing or invalid 'config_params' object");
        }
        if (!metadata_.contains("beam_intrinsics") || !metadata_["beam_intrinsics"].is_object()) {
            throw std::runtime_error("Missing or invalid 'beam_intrinsics' object");
        }
        if (!metadata_.contains("lidar_intrinsics") || !metadata_["lidar_intrinsics"].is_object() ||
            !metadata_["lidar_intrinsics"].contains("lidar_to_sensor_transform")) {
            throw std::runtime_error("Missing or invalid 'lidar_intrinsics.lidar_to_sensor_transform'");
        }

        UDPPROFILELIDAR = metadata_["config_params"]["udp_profile_lidar"].get<std::string>();
        UDPPORTLIDAR = metadata_["config_params"]["udp_port_lidar"].get<uint16_t>();
        UDPDESTINATION = metadata_["config_params"]["udp_dest"].get<std::string>();

    } catch (const std::exception& e) {
        std::cerr << "[Main] Error parsing JSON: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = pipeline::signalHandler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, nullptr);
    sigaction(SIGTERM, &sigIntHandler, nullptr);

#ifdef DEBUG
    std::cout << "[Main] Starting pipeline processes..." << std::endl;
#endif

    // Switch-case based on udp_profile_lidar
    enum class LidarProfile { RNG19_RFL8_SIG16_NIR16, LEGACY, UNKNOWN };
    LidarProfile profile;
    if (UDPPROFILELIDAR == "RNG19_RFL8_SIG16_NIR16") {
        profile = LidarProfile::RNG19_RFL8_SIG16_NIR16;
        LIDARPACKETSIZE = 24832;
    } else if (UDPPROFILELIDAR == "LEGACY") {
        profile = LidarProfile::LEGACY;
        LIDARPACKETSIZE = 24896;
    } else {
        std::cerr << "[Main] Error: Unknown or unsupported udp_profile_lidar." << std::endl;
        return EXIT_FAILURE;
    }

    // ##########################################################################
    // ## CONFIGURE THE UDP SOCKET - UPDATED FOR NEW UDPSOCKET CLASS ##
    // ##########################################################################
    UdpSocketConfig UDPLIDARCONFIG;
    UDPLIDARCONFIG.host = "192.168.75.10";
    UDPLIDARCONFIG.multicastGroup = std::nullopt;
    UDPLIDARCONFIG.localInterfaceIp = "192.168.75.10";
    UDPLIDARCONFIG.port = UDPPORTLIDAR;
    UDPLIDARCONFIG.bufferSize = LIDARPACKETSIZE;
    UDPLIDARCONFIG.receiveTimeout = std::chrono::milliseconds(10000); 
    UDPLIDARCONFIG.reuseAddress = true; 
    UDPLIDARCONFIG.enableBroadcast = false; 
    UDPLIDARCONFIG.ttl =  std::nullopt; 
    // ##########################################################################
    UdpSocketConfig UDPNAVCONFIG;
    UDPNAVCONFIG.host = "192.168.75.10"; 
    UDPNAVCONFIG.port = 6597;     
    UDPNAVCONFIG.multicastGroup = std::nullopt;
    UDPNAVCONFIG.localInterfaceIp = "192.168.75.10";
    UDPNAVCONFIG.bufferSize = 105;
    UDPNAVCONFIG.receiveTimeout = std::chrono::milliseconds(10000);
    UDPNAVCONFIG.reuseAddress = true; 
    UDPNAVCONFIG.enableBroadcast = false;
    UDPNAVCONFIG.ttl = std::nullopt; 
    // ##########################################################################
    
    try {
        std::vector<std::thread> threads;
        boost::asio::io_context IOCTXPOINTS;
        boost::asio::io_context IOCTXNAV;
        switch (profile) {
            case LidarProfile::RNG19_RFL8_SIG16_NIR16:
#ifdef DEBUG
                std::cout << "[Main] Detected RNG19_RFL8_SIG16_NIR16 lidar udp profile." << std::endl;
#endif
                threads.emplace_back([&]() { pipeline.runLidarListenerRng19(IOCTXPOINTS, UDPLIDARCONFIG, std::vector<int>{0}); });
                break;

            case LidarProfile::LEGACY:
#ifdef DEBUG
                std::cout << "[Main] Detected LEGACY lidar udp profile." << std::endl;
#endif
                threads.emplace_back([&]() { pipeline.runLidarListenerLegacy(IOCTXPOINTS, UDPLIDARCONFIG, std::vector<int>{0}); });
                break;

            case LidarProfile::UNKNOWN:
            default:
                std::cerr << "[Main] Error: Unknown or unsupported udp_profile_lidar: " << UDPPROFILELIDAR << std::endl;
                return EXIT_FAILURE;
        }
        
        threads.emplace_back([&]() { pipeline.runNavListener(IOCTXNAV, UDPNAVCONFIG, std::vector<int>{1}); });
        threads.emplace_back([&]() { pipeline.dataAlignment(STEPSIZE, std::vector<int>{2}); });
        threads.emplace_back([&]() { pipeline.sam(std::vector<int>{3,4,5,6}); });
        threads.emplace_back([&]() { pipeline.processLogQueue(log_filename, std::vector<int>{7}); });
        while (pipeline::RUNNING_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        IOCTXPOINTS.stop();
        IOCTXNAV.stop();

        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        std::cout << "[Main] All threads joined. Saving final results..." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: [Main] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "[Main] All processes stopped. Exiting program." << std::endl;
    return EXIT_SUCCESS;
}