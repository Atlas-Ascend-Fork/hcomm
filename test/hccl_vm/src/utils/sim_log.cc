/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_log.h"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <cstring>
#include <limits>
#include <map>
#include <pwd.h>
#include <string>
#include <unistd.h>
#include <errno.h>

#include "sim_common_api.h"
#include "sim_yaml_config.h"
#include "spdlog/details/os-inl.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/stdout_sinks.h"

spdlog::logger* g_logger{nullptr};
namespace hcclvm {
LogWriter g_logWriter = nullptr;
spdlog::level::level_enum g_cannModuleLevels[kLogModuleCount]{};
} // namespace hcclvm

namespace {
// device 侧只在 device_arm 进程内使用：把原生 CANN 配置翻译成现有 logger 的 LogConfig，
// 并提供 CANN 文本格式。日志对象的创建、打印、flush 与释放仍走下面的既有流程。
bool IsDeviceCannLog() { return hcclvm::IsDeviceRole() && hcclvm::IsCannBackend(); }

int ReadCannNumber(const std::string& text, int fallback, int maximum)
{
    int value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && value >= 0 && value <= maximum ?
               value :
               fallback;
}

void ConfigureDeviceCannLog(LogConfig& config, uint32_t deviceKey)
{
    const char* processPath = getenv("ASCEND_PROCESS_LOG_PATH");
    const char* workPath = getenv("ASCEND_WORK_PATH");
    if (processPath && *processPath)
        config.filePath = processPath;
    else if (workPath && *workPath)
        config.filePath = std::string(workPath) + "/log";
    else {
        const passwd* user = getpwuid(getuid());
        if (!user || !user->pw_dir)
            throw spdlog::spdlog_ex("cannot resolve CANN user log directory");
        config.filePath = std::string(user->pw_dir) + "/ascend/log";
    }
    config.filePath += "/run/device-" + std::to_string(deviceKey);
    config.fileBaseName = fmt::format(
        "device-{}_{}", getppid(),
        std::chrono::duration_cast<std::chrono::microseconds>(spdlog::log_clock::now().time_since_epoch()).count());
    config.fileSuffix = ".log";
    config.fileLevel = spdlog::level::trace; // 模块等级在初始化时缓存，文件 sink 接收已过滤记录。
    config.consoleLevel = spdlog::level::off;
    config.maxFileSize = 20 * 1024 * 1024;
    const char* count = getenv("ASCEND_HOST_LOG_FILE_NUM");
    const int files = count ? ReadCannNumber(count, 10, 1000) : 10;
    config.maxFiles = (files > 0 ? files : 10) - 1; // spdlog 参数不包含当前文件
}

class DeviceCannFormatter final : public spdlog::formatter {
public:
    DeviceCannFormatter() : hostPid_(getppid())
    {
        std::ifstream input("/proc/" + std::to_string(hostPid_) + "/comm");
        std::getline(input, hostName_);
    }
    void format(const spdlog::details::log_msg& message, spdlog::memory_buf_t& output) override
    {
        const auto seconds = spdlog::log_clock::to_time_t(message.time);
        const auto local = spdlog::details::os::localtime(seconds);
        char date[32];
        strftime(date, sizeof(date), "%Y-%m-%d-%H:%M:%S", &local);
        const auto micros
            = std::chrono::duration_cast<std::chrono::microseconds>(message.time.time_since_epoch()).count() % 1000000;
        static constexpr const char* levels[] = {"DEBUG", "DEBUG", "INFO", "WARNING", "ERROR", "ERROR", "NULL"};
        const std::string_view payload(message.payload.data(), message.payload.size());
        const auto tagEnd = payload.find(']');
        const auto module
            = tagEnd != std::string_view::npos ? payload.substr(1, tagEnd - 1) : std::string_view("HCCL-VM");
        const auto body = tagEnd != std::string_view::npos ? payload.substr(tagEnd + 1) : payload;
        const char* file = message.source.filename ? message.source.filename : "unknown";
        const char* slash = strrchr(file, '/');
        fmt::format_to(
            std::back_inserter(output), "[{}] {}({},{}):{}.{:03d}.{:03d} [{}:{}] {}\n", levels[message.level], module,
            hostPid_, hostName_, date, micros / 1000, micros % 1000, slash ? slash + 1 : file, message.source.line,
            body);
    }
    std::unique_ptr<spdlog::formatter> clone() const override { return std::make_unique<DeviceCannFormatter>(*this); }

private:
    pid_t hostPid_;
    std::string hostName_;
};

int CannEnvironmentLogLevel(const char* moduleName)
{
    static const auto levels = [] {
        std::map<std::string, int> result;
        const char* global = getenv("ASCEND_GLOBAL_LOG_LEVEL");
        result[""] = global ? ReadCannNumber(global, 3, 4) : 3;
        const char* modules = getenv("ASCEND_MODULE_LOG_LEVEL");
        std::istringstream settings(modules ? modules : "");
        std::string setting;
        while (std::getline(settings, setting, ':')) {
            const auto equal = setting.find('=');
            if (equal == std::string::npos)
                continue;
            const int level = ReadCannNumber(setting.substr(equal + 1), -1, 4);
            if (level >= 0)
                result[setting.substr(0, equal)] = level;
        }
        return result;
    }();
    const auto found = levels.find(moduleName);
    return found == levels.end() ? levels.at("") : found->second;
}

void WriteDeviceCannLog(
    size_t, spdlog::level::level_enum level, spdlog::source_loc source, const char* tag, fmt::string_view format,
    fmt::format_args args)
{
    // 模块等级已由宏在参数求值前比较，不在此重复检查。
    spdlog::memory_buf_t body;
    body.append(tag, tag + std::strlen(tag));
    fmt::vformat_to(std::back_inserter(body), format, args);
    g_logger->log(source, level, spdlog::string_view_t(body.data(), body.size()));
}
} // namespace

std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> InitConsoleSink(const LogConfig& config)
{
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sink->set_level(static_cast<spdlog::level::level_enum>(config.consoleLevel));
    sink->set_pattern("[%l][PID:%P][TID:%t][%s][%!] %v");
    return sink;
}

std::shared_ptr<spdlog::sinks::rotating_file_sink_mt> InitFileSink(const LogConfig& config)
{
    // after close handler
    spdlog::file_event_handlers handlers;
    if (!IsDeviceCannLog())
        handlers.after_close = [config](const std::string& filePath) {
            // 注意: after_close 由 rotating_file_sink_mt 在轮转关闭文件时回调, 此刻该 sink 的
            // mutex_ 仍被持有。若在此处调用 HCCL_VM_*(经 g_logger 回到同一个 file_sink)会重入
            // 同一把非递归 std::mutex, 造成自死锁。因此这里只能直写 stderr, 不得使用日志宏。
            // rename
            static std::atomic<uint32_t> g_log_file_index{0};
            if (filePath.size() < config.fileSuffix.size()
                || filePath.substr(filePath.size() - config.fileSuffix.size()) != config.fileSuffix) {
                fprintf(stderr, "[HCCL-VM][ERROR] Log file name error: %s\n", filePath.c_str());
                return;
            }
            std::ostringstream oss;
            oss << filePath.substr(0, filePath.size() - config.fileSuffix.size()) << "_"
                << std::to_string(g_log_file_index.fetch_add(1, std::memory_order_relaxed)) << config.fileSuffix;
            const std::string newPath = oss.str();
            if (spdlog::details::os::rename(filePath, newPath) != 0) {
                fprintf(
                    stderr, "[HCCL-VM][ERROR] Fail to rename rotating log file: %s -> %s pid %d errno %d: %s\n",
                    filePath.c_str(), newPath.c_str(), getpid(), errno, strerror(errno));
                return;
            }
        };

    std::ostringstream logFileName;
    logFileName << config.filePath << "/" << config.fileBaseName << "_" << std::to_string(getpid())
                << config.fileSuffix;

    auto sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        logFileName.str(), config.maxFileSize, config.maxFiles, !IsDeviceCannLog(), handlers);
    sink->set_level(static_cast<spdlog::level::level_enum>(config.fileLevel));
    if (IsDeviceCannLog())
        sink->set_formatter(std::make_unique<DeviceCannFormatter>());
    else
        sink->set_pattern("[%Y-%m-%d %H:%M:%S.%f][%l][PID:%P][TID:%t][%s:%#][%!] %v");
    return sink;
}

static std::string GetLogYamlConfigPath()
{
    const char* env = std::getenv("HCCL_VM_LOG_CONFIG_PATH");
    if (env && *env) {
        return env;
    }
    return InstallPath::ResolveToInstallRoot("config/log_config.yaml");
}

static int ClampLogLevel(int level)
{
    if (level < 0)
        return 0;
    if (level > 6)
        return 6;
    return level;
}

static int SafeStoi(const std::string& s, int default_val, const char* field_name)
{
    if (s.empty()) {
        std::fprintf(stderr, "[HCCL-VM][WARN] Empty value for '%s', using default %d\n", field_name, default_val);
        return default_val;
    }
    try {
        size_t pos = 0;
        long val = std::stol(s, &pos);
        if (pos != s.size()) {
            std::fprintf(
                stderr, "[HCCL-VM][WARN] Invalid value for '%s': '%s', using default %d\n", field_name, s.c_str(),
                default_val);
            return default_val;
        }
        if (val < static_cast<long>(std::numeric_limits<int>::min())
            || val > static_cast<long>(std::numeric_limits<int>::max())) {
            std::fprintf(
                stderr, "[HCCL-VM][WARN] Out of range value for '%s': '%s', using default %d\n", field_name, s.c_str(),
                default_val);
            return default_val;
        }
        return static_cast<int>(val);
    } catch (const std::invalid_argument&) {
        std::fprintf(
            stderr, "[HCCL-VM][WARN] Non-numeric value for '%s': '%s', using default %d\n", field_name, s.c_str(),
            default_val);
        return default_val;
    } catch (const std::out_of_range&) {
        std::fprintf(
            stderr, "[HCCL-VM][WARN] Out of range value for '%s': '%s', using default %d\n", field_name, s.c_str(),
            default_val);
        return default_val;
    }
}

static size_t SafeStoull(const std::string& s, size_t default_val, const char* field_name)
{
    if (s.empty()) {
        std::fprintf(stderr, "[HCCL-VM][WARN] Empty value for '%s', using default %zu\n", field_name, default_val);
        return default_val;
    }
    try {
        size_t pos = 0;
        unsigned long long val = std::stoull(s, &pos);
        if (pos != s.size()) {
            std::fprintf(
                stderr, "[HCCL-VM][WARN] Invalid value for '%s': '%s', using default %zu\n", field_name, s.c_str(),
                default_val);
            return default_val;
        }
        return static_cast<size_t>(val);
    } catch (const std::invalid_argument&) {
        std::fprintf(
            stderr, "[HCCL-VM][WARN] Non-numeric value for '%s': '%s', using default %zu\n", field_name, s.c_str(),
            default_val);
        return default_val;
    } catch (const std::out_of_range&) {
        std::fprintf(
            stderr, "[HCCL-VM][WARN] Out of range value for '%s': '%s', using default %zu\n", field_name, s.c_str(),
            default_val);
        return default_val;
    }
}

LogConfig LoadLogConfig(const std::string& process_name)
{
    LogConfig cfg;
    // 该配置只属于既有 stub 日志流程；cann 模式不读 YAML、不派生等级或路径。
    if (hcclvm::IsCannBackend())
        return cfg;

    static const std::map<std::string, std::string> yaml_node_map = {{"device_aarch64", "proxy"}};
    std::string node_name = process_name;
    auto it = yaml_node_map.find(process_name);
    if (it != yaml_node_map.end()) {
        node_name = it->second;
    }

    std::map<std::string, std::string> fields;
    if (LoadYamlStringMap(GetLogYamlConfigPath(), node_name, fields)) {
        auto it_console = fields.find("console_level");
        auto it_file = fields.find("file_level");
        auto it_max_size = fields.find("max_file_size");
        auto it_max_num = fields.find("max_files");
        auto it_path = fields.find("file_path");
        auto it_suffix = fields.find("file_suffix");
        auto it_compress = fields.find("enable_compress");

        if (it_console != fields.end()) {
            cfg.consoleLevel = ClampLogLevel(SafeStoi(it_console->second, LogConfig{}.consoleLevel, "console_level"));
        }
        if (it_file != fields.end()) {
            cfg.fileLevel = ClampLogLevel(SafeStoi(it_file->second, LogConfig{}.fileLevel, "file_level"));
        }
        if (it_max_size != fields.end()) {
            cfg.maxFileSize = SafeStoull(it_max_size->second, LogConfig{}.maxFileSize, "max_file_size");
        }
        if (it_max_num != fields.end()) {
            cfg.maxFiles = SafeStoull(it_max_num->second, LogConfig{}.maxFiles, "max_files");
        }
        if (it_path != fields.end()) {
            cfg.filePath = InstallPath::ResolveToInstallRoot(it_path->second);
        }
        if (it_suffix != fields.end()) {
            cfg.fileSuffix = it_suffix->second;
        }
        if (it_compress != fields.end()) {
            cfg.enableCompress = (it_compress->second == "true" || it_compress->second == "1");
        }
    } else {
        if (process_name == "proxy") {
            cfg.filePath = InstallPath::ResolveToInstallRoot("logs/proxy");
        } else if (process_name == "device_aarch64") {
            cfg.filePath = InstallPath::ResolveToInstallRoot("logs/proxy");
        } else if (process_name == "runner") {
            cfg.filePath = InstallPath::ResolveToInstallRoot("logs/runner");
        } else if (process_name == "checker") {
            cfg.filePath = InstallPath::ResolveToInstallRoot("logs/checker");
        } else if (process_name == "hccl_vm") {
            cfg.filePath = InstallPath::ResolveToInstallRoot("logs");
        }
    }

    if (process_name == "device_aarch64") {
        cfg.fileBaseName = "proxy_" + std::to_string(getppid());
    } else {
        cfg.fileBaseName = process_name;
    }
    return cfg;
}

void InitLogger(const LogConfig& config, uint32_t deviceKey)
{
    const bool cann = hcclvm::IsCannBackend();
    const bool device = hcclvm::IsDeviceRole();
    // host cann 在入口前绑定写入函数；后续 InitLogger 不重建或覆盖原生配置。
    if (cann && !device && hcclvm::g_logWriter)
        return;
    DeInitLogger();
    try {
        if (cann) {
            constexpr spdlog::level::level_enum levels[] = {
                spdlog::level::trace, spdlog::level::info, spdlog::level::warn, spdlog::level::err, spdlog::level::off};
            for (size_t i = 0; i < hcclvm::kLogModuleCount; ++i) {
                hcclvm::g_cannModuleLevels[i] = levels[CannEnvironmentLogLevel(hcclvm::kCannLogModules[i].name)];
            }
        }
        if (cann && !device) {
            // 不创建 spdlog 对象、不查询真库等级；宏前置过滤后直接转发。
            hcclvm::g_logWriter = hcclvm::InitializeCannLog();
        } else {
            std::unique_ptr<spdlog::logger> logger;
            if (cann) {
                auto deviceConfig = config;
                ConfigureDeviceCannLog(deviceConfig, deviceKey);
                logger = std::make_unique<spdlog::logger>("SLOG", InitFileSink(deviceConfig));
                logger->set_level(spdlog::level::trace);
            } else {
                auto consoleSink = InitConsoleSink(config);
                auto fileSink = InitFileSink(config);
                logger
                    = std::make_unique<spdlog::logger>("muti-logger", spdlog::sinks_init_list({consoleSink, fileSink}));
                // 保持原 stub 行为：logger 全放行，控制台/文件 sink 各自过滤。
                logger->set_level(spdlog::level::trace);
            }
            logger->flush_on(spdlog::level::warn);
            g_logger = logger.release();
            if (cann)
                hcclvm::g_logWriter = WriteDeviceCannLog;
        }
        static const int registered = std::atexit(DeInitLogger);
        (void)registered;
    } catch (const std::exception& error) {
        DeInitLogger();
        std::fprintf(stderr, "[HCCL-VM][ERROR] Logger init failed: %s\n", error.what());
    }
}

void FlushLog()
{
    if (g_logger)
        g_logger->flush();
    else if (hcclvm::g_logWriter)
        hcclvm::FlushCannLog();
}

void DeInitLogger()
{
    FlushLog();
    hcclvm::g_logWriter = nullptr;
    delete g_logger;
    g_logger = nullptr;
}

namespace {
// host proxy 的首条 HCCL_VM 日志可能早于任何 DlogRecord/InitLogger 调用。
// cann 在入口前完成绑定，不在日志热路径添加惰性初始化；device 等 argv 主键就绪再初始化。
struct HostCannLogStartup {
    HostCannLogStartup()
    {
        if (!hcclvm::IsDeviceRole() && hcclvm::IsCannBackend())
            InitLogger(LogConfig{});
    }
} hostCannLogStartup;
} // namespace
