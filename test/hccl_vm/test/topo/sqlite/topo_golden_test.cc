/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * 集群拓扑生成 UT (golden基线比对)
 *
 * 覆盖对象:
 *   1. asset/generate_cluster_topo.sh / asset/generate_server_topo.sh / asset/allocate_eid.sh 脚本:
 *      - 以 asset/cluster_model/config/cluster 下配置文件为输入, 运行 generate_cluster_topo.sh
 *      - 生成的 topo.json / hccl_rootinfo.json / servers_info.json 与 test/topo/golden 下基线文件比对
 *   2. src/topo 代码(mock-comm命令链路):
 *      - 以 asset/cluster_model/topo_meta 下配置为 mock-comm 用例
 *      - 调用 AscendClusterTopoParser::InitClusterTopo + InitCommunicationDomain 生成 ranktable.json
 *      - 生成的 ranktable.json 与 test/topo/golden/ranktable 下基线文件比对
 *
 * 用例配置: test/topo/topo_cases.json
 * 基线刷新: TOPO_UT_UPDATE_GOLDEN=1 运行本测试(或执行 test/topo/generate_golden.sh),
 *           会以当前代码/脚本的输出重建 test/topo/golden 目录。
 *           注意: 更新模式会先清空整个golden目录, 必须全量运行, 不能带 --gtest_filter。
 *
 * 说明:
 *   - 脚本以"安装态布局"(script/ + config/)在临时目录中软链接运行, 与安装目录行为一致;
 *   - 同一集群内各server的topo.json完全相同, hccl_rootinfo.json仅EID(IP)信息不同,
 *     因此集群golden仅保留 superpod0/server0 的 topo.json / hccl_rootinfo.json,
 *     另加全集群级的 servers_info.json(校验allocate_eid.sh对全部server的IP分配);
 *     POD形态(4层拓扑)下topo.json为POD级单份(superpod0/server0/topo.json,
 *     local_id为POD全局id = board_id*每board卡数+board内物理id), rootinfo仍每board一份,
 *     golden保留首个与末个board的hccl_rootinfo.json;
 *   - hccl_rootinfo.json 中的 topo_file_path 含绝对路径(依赖临时目录), 写golden前统一归一化;
 *   - JSON 采用语义比对(解析后逐节点比对), 失败时输出首个差异节点的路径与两侧取值。
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "cmd_cluster_model_utils.h"
#include "sim_common_api.h"
#include "topo_ascend_cluster_parser.h"
#include "storage/storage_runtime.h"
#include "storage_composition/storage_composition.h"
#include "runtime_state/db_sim_runner_ops.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
constexpr int kMaxDiffReport = 8; // 单文件最多报告的差异节点数

// 等价迁移前 Sqlite 单库 SetDbPath + ClearAll() 的语义：
// 关闭进程存储会话、删除数据库文件（含 WAL/SHM）后，以测试私有路径重建
// RUNNER_DB + OP_DATA_DB 进程会话（每个用例相互独立）。
bool ResetSimulatorStorage(const fs::path& workDir)
{
    const std::string runnerPath = (workDir / "hccl_sim.db").string();
    const std::string opDataPath = (workDir / "hccl_sim_op.db").string();

    (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
    for (const std::string* path : {&runnerPath, &opDataPath}) {
        std::remove(path->c_str());
        std::remove((*path + "-wal").c_str());
        std::remove((*path + "-shm").c_str());
    }

    HcclSim::Storage::StorageRuntimeConfig config;
    config.databases.push_back({std::string{"RUNNER_DB"}, "sqlite", runnerPath});
    config.databases.push_back({std::string{"OP_DATA_DB"}, "sqlite", opDataPath});
    return sim::EnsureSimulatorStorageSession(config).ok();
}

// ------------------------------------------------------------------
// 工具函数
// ------------------------------------------------------------------
std::string GetExePath()
{
    char buf[4096] = {0};
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) {
        return "";
    }
    buf[len] = '\0';
    return std::string(buf);
}

// 从测试二进制位置(build/output/bin)向上回溯, 定位仓库根目录(包含 asset/ 与 test/ 的目录)
std::string FindRepoRoot()
{
    fs::path cur = fs::path(GetExePath()).parent_path();
    for (int i = 0; i < 10; ++i) {
        if (fs::exists(cur / "asset" / "generate_cluster_topo.sh")
            && fs::exists(cur / "test" / "topo" / "topo_cases.json")) {
            return cur.string();
        }
        if (!cur.has_parent_path() || cur.parent_path() == cur) {
            break;
        }
        cur = cur.parent_path();
    }
    return "";
}

bool LoadJsonFile(const std::string& path, json& out, std::string& err)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        err = "cannot open file: " + path;
        return false;
    }
    try {
        ifs >> out;
    } catch (const json::exception& e) {
        err = "json parse error in " + path + ": " + e.what();
        return false;
    }
    return true;
}

// hccl_rootinfo.json 的 topo_file_path 为绝对路径(依赖运行时临时目录), 归一化为占位符后再比对。
// 叶子节点目录: server形态 = superpod{N}/server{M}/; POD形态(4层拓扑) = superpod{N}/server{M}/board{B}/
void NormalizePathFields(json& node)
{
    static const std::regex pathRe("^.*/(superpod\\d+/server\\d+/(?:board\\d+/)?topo\\.json)$");
    if (node.is_object()) {
        for (auto it = node.begin(); it != node.end(); ++it) {
            if (it.key() == "topo_file_path" && it.value().is_string()) {
                std::string value = it.value().get<std::string>();
                std::smatch match;
                if (std::regex_search(value, match, pathRe)) {
                    it.value() = "<CLUSTER_DIR>/" + match[1].str();
                }
                continue;
            }
            NormalizePathFields(it.value());
        }
    } else if (node.is_array()) {
        for (auto& item : node) {
            NormalizePathFields(item);
        }
    }
}

// 递归语义比对, 记录差异节点(JSON路径 + 两端取值)
void DiffJson(const json& golden, const json& actual, const std::string& path, std::vector<std::string>& diffs)
{
    if (diffs.size() >= kMaxDiffReport) {
        return;
    }
    if (golden.type() != actual.type()) {
        diffs.push_back(path + ": type mismatch, golden=" + golden.type_name() + ", actual=" + actual.type_name());
        return;
    }
    if (golden.is_object()) {
        std::set<std::string> keys;
        for (auto it = golden.begin(); it != golden.end(); ++it) {
            keys.insert(it.key());
        }
        for (auto it = actual.begin(); it != actual.end(); ++it) {
            keys.insert(it.key());
        }
        for (const auto& key : keys) {
            std::string child = path + "/" + key;
            if (!golden.contains(key)) {
                diffs.push_back(child + ": only in actual, value=" + actual[key].dump());
            } else if (!actual.contains(key)) {
                diffs.push_back(child + ": only in golden, value=" + golden[key].dump());
            } else {
                DiffJson(golden[key], actual[key], child, diffs);
            }
            if (diffs.size() >= kMaxDiffReport) {
                return;
            }
        }
        return;
    }
    if (golden.is_array()) {
        if (golden.size() != actual.size()) {
            diffs.push_back(
                path + ": size mismatch, golden=" + std::to_string(golden.size())
                + ", actual=" + std::to_string(actual.size()));
            return;
        }
        for (size_t i = 0; i < golden.size(); ++i) {
            DiffJson(golden[i], actual[i], path + "[" + std::to_string(i) + "]", diffs);
            if (diffs.size() >= kMaxDiffReport) {
                return;
            }
        }
        return;
    }
    if (golden != actual) {
        diffs.push_back(path + ": golden=" + golden.dump() + ", actual=" + actual.dump());
    }
}

// ------------------------------------------------------------------
// 测试全局环境: 临时工作目录 + 安装态布局 + 用例配置
// ------------------------------------------------------------------
class TopoUtEnv {
public:
    struct ClusterCase {
        std::string cluster;
    };
    struct MockCommCase {
        std::string name;
        std::string cluster;
        std::string topoMeta;
    };

    static TopoUtEnv& Instance()
    {
        static TopoUtEnv env;
        return env;
    }

    // 完整初始化(临时目录/软链接布局/DB路径), 每个用例SetUp中调用
    // 用例列表在构造阶段加载(供测试注册使用), 其余资源首次调用本函数时初始化
    bool Init(std::string& err)
    {
        if (inited_) {
            return initOk_;
        }
        inited_ = true;

        if (!casesLoaded_) {
            err = casesLoadErr_.empty() ? "topo_cases.json not loaded" : casesLoadErr_;
            initOk_ = false;
            return false;
        }

        // 更新模式: 先清空旧基线目录, 避免残留已下线用例的golden文件
        if (UpdateGolden()) {
            std::cout << "[golden] update mode: rebuilding golden dirs (do NOT use --gtest_filter)" << std::endl;
            std::error_code ec;
            fs::path goldenBase = fs::path(repoRoot_) / "test" / "topo" / "golden";
            fs::remove_all(goldenBase / "cluster", ec);
            fs::remove_all(goldenBase / "ranktable", ec);
        }

        workDir_ = fs::temp_directory_path() / ("hccl_vm_topo_ut_" + std::to_string(static_cast<long>(getpid())));
        std::error_code ec;
        fs::remove_all(workDir_, ec);
        fs::create_directories(workDir_, ec);
        if (ec) {
            err = "cannot create work dir " + workDir_.string() + ": " + ec.message();
            initOk_ = false;
            return false;
        }

        if (!SetupStage(err)) {
            initOk_ = false;
            return false;
        }

        // mock-comm 链路读取 install root 下 config/topo_meta; 数据库使用独立临时文件
        if (!ResetSimulatorStorage(workDir_)) {
            err = "cannot init simulator storage session";
            initOk_ = false;
            return false;
        }
        initOk_ = true;
        return true;
    }

    bool UpdateGolden() const
    {
        const char* env = std::getenv("TOPO_UT_UPDATE_GOLDEN");
        return env != nullptr && std::string(env) == "1";
    }

    const std::string& RepoRoot() const { return repoRoot_; }
    const fs::path& WorkDir() const { return workDir_; }
    const std::vector<ClusterCase>& ClusterCases() const { return clusterCases_; }
    const std::vector<MockCommCase>& MockCommCases() const { return mockCommCases_; }

    // 懒生成: 按需运行 generate_cluster_topo.sh, 返回 <work>/out/<cluster> 目录
    fs::path GetClusterDir(const std::string& cluster, std::string& err)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = generated_.find(cluster);
        if (it != generated_.end()) {
            return it->second;
        }

        fs::path clusterDir = workDir_ / "out" / cluster;
        std::error_code ec;
        fs::remove_all(clusterDir, ec);

        fs::path logPath = workDir_ / ("gen_" + cluster + ".log");
        std::string cmd = "bash \"" + (stageDir_ / "script" / "generate_cluster_topo.sh").string() + "\" -o \""
                          + (workDir_ / "out").string() + "\" " + cluster + ".yaml > \"" + logPath.string() + "\" 2>&1";
        int status = std::system(cmd.c_str());
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            std::ifstream log(logPath);
            std::string tail;
            std::string line;
            while (std::getline(log, line)) {
                tail = line + "\n" + tail;
                if (tail.size() > 4000) {
                    break;
                }
            }
            err = "generate_cluster_topo.sh failed for " + cluster + " (exit status=" + std::to_string(status)
                  + "), log tail:\n" + tail;
            return fs::path();
        }
        if (!fs::exists(clusterDir)) {
            err = "cluster output dir not found after generation: " + clusterDir.string();
            return fs::path();
        }
        generated_[cluster] = clusterDir;
        return clusterDir;
    }

private:
    TopoUtEnv()
    {
        repoRoot_ = FindRepoRoot();
        casesLoaded_ = LoadCases(casesLoadErr_);
    }
    ~TopoUtEnv()
    {
        std::error_code ec;
        fs::remove_all(workDir_, ec);
    }

    // 解析用例配置(仅依赖仓库根目录定位, 可在测试注册阶段调用)
    bool LoadCases(std::string& err)
    {
        if (repoRoot_.empty()) {
            err = "cannot locate repo root from test binary path (expect build/output/bin)";
            return false;
        }
        std::string caseFile = repoRoot_ + "/test/topo/topo_cases.json";
        if (!LoadJsonFile(caseFile, cases_, err)) {
            return false;
        }
        try {
            for (const auto& c : cases_.at("cluster_cases")) {
                clusterCases_.push_back(ClusterCase{c.at("cluster").get<std::string>()});
            }
            for (const auto& c : cases_.at("mock_comm_cases")) {
                MockCommCase mc;
                mc.name = c.at("name").get<std::string>();
                mc.cluster = c.at("cluster").get<std::string>();
                mc.topoMeta = c.at("topo_meta").get<std::string>();
                mockCommCases_.push_back(mc);
            }
        } catch (const json::exception& e) {
            err = std::string("invalid topo_cases.json: ") + e.what();
            return false;
        }
        if (clusterCases_.empty() || mockCommCases_.empty()) {
            err = "topo_cases.json has empty cluster_cases/mock_comm_cases";
            return false;
        }
        return true;
    }

    // 搭建与安装目录一致的布局: script/ 下软链接仓库 asset 脚本, config/ 下软链接集群配置
    bool SetupStage(std::string& err)
    {
        stageDir_ = workDir_ / "stage";
        fs::path scriptDir = stageDir_ / "script";
        fs::path configDir = stageDir_ / "config";
        std::error_code ec;
        fs::create_directories(scriptDir, ec);
        fs::create_directories(configDir, ec);
        if (ec) {
            err = "cannot create stage dirs: " + ec.message();
            return false;
        }

        fs::path assetDir = fs::path(repoRoot_) / "asset";
        for (const auto& script : {"generate_cluster_topo.sh", "generate_server_topo.sh", "allocate_eid.sh"}) {
            fs::path link = scriptDir / script;
            fs::remove(link, ec);
            fs::create_symlink(assetDir / script, link, ec);
            if (ec || !fs::exists(link)) {
                err = "cannot create symlink " + link.string() + ": " + ec.message();
                return false;
            }
        }

        auto linkConfig = [&](const std::string& name) {
            fs::path link = configDir / name;
            fs::remove(link, ec);
            fs::create_symlink(assetDir / "cluster_model" / "config" / name, link, ec);
            if (ec || !fs::exists(link)) {
                err = "cannot create symlink " + link.string() + ": " + ec.message();
                return false;
            }
            return true;
        };
        if (!linkConfig("cluster") || !linkConfig("server_or_pod")) {
            return false;
        }
        return true;
    }

    bool inited_{false};
    bool initOk_{false};
    bool casesLoaded_{false};
    std::string casesLoadErr_;
    std::string repoRoot_;
    fs::path workDir_;
    fs::path stageDir_;
    json cases_;
    std::vector<ClusterCase> clusterCases_;
    std::vector<MockCommCase> mockCommCases_;
    std::map<std::string, fs::path> generated_;
    std::mutex mutex_;
};

// ------------------------------------------------------------------
// golden 读写
// ------------------------------------------------------------------
fs::path GoldenClusterFile(const std::string& cluster, const std::string& relFile)
{
    return fs::path(TopoUtEnv::Instance().RepoRoot()) / "test" / "topo" / "golden" / "cluster" / cluster / relFile;
}

fs::path GoldenRanktableFile(const std::string& caseName)
{
    return fs::path(TopoUtEnv::Instance().RepoRoot()) / "test" / "topo" / "golden" / "ranktable" / (caseName + ".json");
}

// golden缺失时, 提示刷新基线
std::string GoldenHint()
{
    return "golden file missing or outdated, regenerate by: TOPO_UT_UPDATE_GOLDEN=1 "
           "./test_topo_golden (or bash test/topo/generate_golden.sh)";
}

// 比较或写入golden; 返回true表示通过(或已更新基线)
// 写基线时先做路径归一化, 保证golden文件不含机器相关的绝对路径, 可跨环境使用
bool CompareOrWriteGolden(const fs::path& actualFile, const fs::path& goldenFile)
{
    std::string err;
    json actual;
    if (!LoadJsonFile(actualFile.string(), actual, err)) {
        ADD_FAILURE() << err;
        return false;
    }
    NormalizePathFields(actual);

    if (TopoUtEnv::Instance().UpdateGolden()) {
        std::error_code ec;
        fs::create_directories(goldenFile.parent_path(), ec);
        if (ec) {
            ADD_FAILURE() << "cannot create golden dir " << goldenFile.parent_path() << ": " << ec.message();
            return false;
        }
        std::ofstream ofs(goldenFile, std::ios::trunc);
        if (!ofs.is_open()) {
            ADD_FAILURE() << "cannot write golden " << goldenFile;
            return false;
        }
        ofs << actual.dump(4) << std::endl;
        std::cout << "[golden] updated: " << goldenFile << std::endl;
        return true;
    }

    if (!fs::exists(goldenFile)) {
        ADD_FAILURE() << GoldenHint() << "\n  missing golden: " << goldenFile;
        return false;
    }

    json golden;
    if (!LoadJsonFile(goldenFile.string(), golden, err)) {
        ADD_FAILURE() << err;
        return false;
    }

    std::vector<std::string> diffs;
    DiffJson(golden, actual, "", diffs);
    if (!diffs.empty()) {
        ADD_FAILURE() << "golden mismatch: " << actualFile.string() << "\n  golden: " << goldenFile
                      << "\n  actual: " << actualFile << "\n  diffs (first " << diffs.size() << "):";
        for (const auto& d : diffs) {
            std::cout << "    " << d << std::endl;
        }
        return false;
    }
    return true;
}
} // namespace

// ------------------------------------------------------------------
// Part 0: 环境自检(仓库根目录/用例配置/脚本布局可用)
// ------------------------------------------------------------------
TEST(TopoGoldenEnvTest, EnvironmentReady)
{
    std::string err;
    ASSERT_TRUE(TopoUtEnv::Instance().Init(err)) << err;
    EXPECT_FALSE(TopoUtEnv::Instance().ClusterCases().empty());
    EXPECT_FALSE(TopoUtEnv::Instance().MockCommCases().empty());
}

// ------------------------------------------------------------------
// Part 1: 集群拓扑生成脚本 golden 比对 (generate_cluster_topo.sh 及其调用链)
// ------------------------------------------------------------------
class ClusterTopoGoldenTest : public testing::TestWithParam<TopoUtEnv::ClusterCase> {
protected:
    void SetUp() override
    {
        std::string err;
        ASSERT_TRUE(TopoUtEnv::Instance().Init(err)) << err;
    }
};

TEST_P(ClusterTopoGoldenTest, ClusterTopoMatchGolden)
{
    const auto& cc = GetParam();
    std::string err;
    fs::path clusterDir = TopoUtEnv::Instance().GetClusterDir(cc.cluster, err);
    ASSERT_FALSE(clusterDir.empty()) << err;

    // 同一集群内各叶子节点的拓扑完全相同, hccl_rootinfo.json仅EID/IP信息不同,
    // 因此只校验首个叶子节点(外加POD形态的末个board, 用于捕获board序号/IP的CC段映射错误);
    // servers_info.json为集群级文件, 覆盖allocate_eid.sh对全部叶子节点的IP分配, 需要完整校验。
    // 产物布局: server形态 = superpod0/server0/{topo.json,hccl_rootinfo.json};
    //           POD形态(4层拓扑) = superpod0/server0/topo.json(POD级单份, local_id为POD全局id)
    //                              + superpod0/server0/board{b}/hccl_rootinfo.json(每board一份)
    std::vector<std::string> relFiles = {"servers_info.json"};
    const fs::path srv0Dir = clusterDir / "superpod0" / "server0";
    std::vector<int> boardIds;
    if (fs::is_directory(srv0Dir)) {
        for (const auto& entry : fs::directory_iterator(srv0Dir)) {
            if (!entry.is_directory()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind("board", 0) == 0 && name.size() > 5) {
                const std::string numStr = name.substr(5);
                bool allDigits = true;
                for (char c : numStr) {
                    if (!std::isdigit(static_cast<unsigned char>(c))) {
                        allDigits = false;
                        break;
                    }
                }
                if (allDigits) {
                    boardIds.push_back(std::stoi(numStr));
                }
            }
        }
    }
    std::sort(boardIds.begin(), boardIds.end());
    relFiles.push_back("superpod0/server0/topo.json");
    if (boardIds.empty()) {
        relFiles.push_back("superpod0/server0/hccl_rootinfo.json");
    } else {
        // 首个board + 末个board(末个board的IP CC段/net_instance_id中的board序号最易出错)
        std::set<int> picked = {boardIds.front(), boardIds.back()};
        for (int boardId : picked) {
            relFiles.push_back("superpod0/server0/board" + std::to_string(boardId) + "/hccl_rootinfo.json");
        }
    }

    for (const auto& rel : relFiles) {
        ASSERT_TRUE(fs::exists(clusterDir / rel)) << "expected file not generated: " << (clusterDir / rel).string();
    }

    int failed = 0;
    for (const auto& rel : relFiles) {
        if (!CompareOrWriteGolden(clusterDir / rel, GoldenClusterFile(cc.cluster, rel))) {
            failed++;
        }
    }
    EXPECT_EQ(failed, 0) << failed << "/" << relFiles.size() << " files mismatch for cluster " << cc.cluster;
}

INSTANTIATE_TEST_SUITE_P(TopoGolden, ClusterTopoGoldenTest, ::testing::ValuesIn(TopoUtEnv::Instance().ClusterCases()));

// ------------------------------------------------------------------
// Part 2: mock-comm 链路 golden 比对 (src/topo: InitClusterTopo + InitCommunicationDomain)
// ------------------------------------------------------------------
class MockCommGoldenTest : public testing::TestWithParam<TopoUtEnv::MockCommCase> {
protected:
    void SetUp() override
    {
        std::string err;
        ASSERT_TRUE(TopoUtEnv::Instance().Init(err)) << err;

        // 每个用例前清理通信域与DB表, 保证用例间相互独立
        AscendClusterTopoParser::GetInstance().SetClusterStatus(HvmClusterStatus::COMM_DOMAIN_UNINIT);
        ASSERT_TRUE(ResetSimulatorStorage(TopoUtEnv::Instance().WorkDir()));
    }
};

TEST_P(MockCommGoldenTest, RanktableMatchGolden)
{
    const auto& mc = GetParam();
    std::string err;
    fs::path clusterDir = TopoUtEnv::Instance().GetClusterDir(mc.cluster, err);
    ASSERT_FALSE(clusterDir.empty()) << err;

    // 链路与 hccl-vm mock-comm <topo_meta> 一致: 解析集群静态拓扑 + topo_meta 初始化通信域
    ASSERT_EQ(
        AscendClusterTopoParser::GetInstance().InitClusterTopo(clusterDir.string()), HcclVmResult::HCCL_SIM_SUCCESS)
        << "InitClusterTopo failed for " << mc.cluster;

    TopoMeta topoMeta;
    ASSERT_TRUE(ParseYamlTopo(mc.topoMeta, topoMeta)) << "ParseYamlTopo failed for topo_meta " << mc.topoMeta;

    ASSERT_EQ(
        AscendClusterTopoParser::GetInstance().InitCommunicationDomain(topoMeta, false), HcclVmResult::HCCL_SIM_SUCCESS)
        << "InitCommunicationDomain failed for " << mc.topoMeta << " on " << mc.cluster;

    fs::path ranktable = fs::path(InstallPath::ResolveToInstallRoot("data/ranktable.json"));
    ASSERT_TRUE(fs::exists(ranktable)) << "ranktable.json not generated: " << ranktable;

    EXPECT_TRUE(CompareOrWriteGolden(ranktable, GoldenRanktableFile(mc.name)));
}

INSTANTIATE_TEST_SUITE_P(TopoGolden, MockCommGoldenTest, ::testing::ValuesIn(TopoUtEnv::Instance().MockCommCases()));

// ------------------------------------------------------------------
// Part 3: UBOE端口标记校验 (server topo yaml port/port_group protocols=UBOE → EndPoint.is_uboe)
// ------------------------------------------------------------------
// 以 ascend950_server_topo_normal_uboe.yaml 生成的集群为输入:
//   - port_group layer2 protocols=["UBOE"], ports=["1/8"] → 对应EndPoint is_uboe=true
//   - 其余端口(layer0 P2P / layer1 PG / layer3 d2h) → is_uboe=false
// EID 87~86位为die id, 85~80位为端口号(见EID地址分配规则); UBOE端口1/8 → eid[5]=0x48
TEST(TopoUboeEndPointTest, UboePortEndPointMarked)
{
    std::string err;
    ASSERT_TRUE(TopoUtEnv::Instance().Init(err)) << err;
    fs::path clusterDir = TopoUtEnv::Instance().GetClusterDir("ascend950_cluster_4_server_normal_uboe", err);
    ASSERT_FALSE(clusterDir.empty()) << err;

    AscendClusterTopoParser::GetInstance().SetClusterStatus(HvmClusterStatus::COMM_DOMAIN_UNINIT);
    // 独立临时库重置存储会话, 保证本用例 EndPoint 表只含本次 UBOE 集群数据
    fs::path uboeWorkDir
        = fs::temp_directory_path() / ("hccl_vm_topo_ut_uboe_" + std::to_string(static_cast<long>(getpid())));
    std::error_code ec;
    fs::remove_all(uboeWorkDir, ec);
    fs::create_directories(uboeWorkDir, ec);
    ASSERT_TRUE(ResetSimulatorStorage(uboeWorkDir)) << "reset simulator storage failed";
    ASSERT_EQ(
        AscendClusterTopoParser::GetInstance().InitClusterTopo(clusterDir.string()), HcclVmResult::HCCL_SIM_SUCCESS)
        << "InitClusterTopo failed for ascend950_cluster_4_server_normal_uboe";

    auto endPointsResult
        = sim::runtime::Db::GetByPred<sim::runtime::EndPoint>(HcclSim::Storage::All<sim::runtime::EndPoint>());
    auto endPoints = endPointsResult.value_or(std::vector<sim::runtime::EndPoint>{});
    ASSERT_FALSE(endPoints.empty());

    // UBOE端口识别: 不能按EID高位字段判断 —— 按EID分配规则"注意"项, UBOE协议端口hcomm业务
    // 强制使用IPV4地址, 其EID高位(superpod段/die id/端口号)全为0(纯v4映射形态),
    // 与非UBOE的die0/port0端口EID高位相同, 无法区分。
    // 故改为按端口名判定: 该拓扑中UBOE端口为port_group {layer:2, protocols:["UBOE"], ports:["1/8"]}
    const std::string kUboePortName = "1/8";
    std::map<uint64_t, std::string> endpointToPortName;
    auto mappingsResult = sim::runtime::Db::GetByPred<sim::runtime::EndPointPortMapping>(
        HcclSim::Storage::All<sim::runtime::EndPointPortMapping>());
    for (const auto& mapping : mappingsResult.value_or(std::vector<sim::runtime::EndPointPortMapping>{})) {
        auto portResult = sim::runtime::Db::GetById<sim::runtime::Port>(mapping.port_id);
        if (portResult.ok()) {
            endpointToPortName[mapping.endpoint_id] = std::string(portResult->name);
        }
    }

    uint32_t uboeCnt = 0;
    for (const auto& ep : endPoints) {
        const auto nameIt = endpointToPortName.find(ep.id);
        const bool isUboePort = (nameIt != endpointToPortName.end() && nameIt->second == kUboePortName);
        EXPECT_EQ(ep.is_uboe != 0, isUboePort)
            << "endpoint id=" << ep.id << " port=" << (nameIt == endpointToPortName.end() ? "<none>" : nameIt->second)
            << " is_uboe=" << static_cast<int>(ep.is_uboe);
        if (ep.is_uboe != 0) {
            uboeCnt++;
            // UBOE端口EID为纯v4映射形态: 高12字节全0, 低32位为IPv4地址
            bool highBytesZero = true;
            for (size_t i = 0; i < 12; ++i) {
                if (ep.eid[i] != 0) {
                    highBytesZero = false;
                    break;
                }
            }
            EXPECT_TRUE(highBytesZero) << "UBOE endpoint id=" << ep.id << " eid high bytes are not all zero";
        }
    }
    // 4 server × 8 device, 每device 1个UBOE端口(1/8)
    EXPECT_EQ(uboeCnt, 32u);
}

// ------------------------------------------------------------------
// Part 4: POD形态(4层拓扑)的board层落库与ranktable校验
//   校验点:
//     1. 每个board一行Server(board_id正确) + 一行Host(host IP的CC段 == board_id)
//     2. Device.physical_id为board内物理id([0,7]), 而非POD内全局id([0,63])
//     3. mock-comm后 Server.used_dev_num == board内卡数(8), 即业务侧aclrtGetDeviceCount()返回8
//     4. ranktable.json中device_id为board内物理id, local_id为POD全局id
//        (= board_id*8 + device_id, 与POD级topo.json的local_id值域一致),
//        且layer0的net_instance_id每board唯一
// ------------------------------------------------------------------
namespace {
// 从点分IP字符串取第idx个八位组(0=AA,1=BB,2=CC,3=DD); 解析失败返回-1
int IpOctet(const std::string& ip, int idx)
{
    std::vector<std::string> parts;
    std::string cur;
    for (char c : ip) {
        if (c == '.') {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    parts.push_back(cur);
    if (idx < 0 || idx >= static_cast<int>(parts.size())) {
        return -1;
    }
    try {
        return std::stoi(parts[static_cast<size_t>(idx)]);
    } catch (...) {
        return -1;
    }
}
} // namespace

TEST(TopoPodFormBoardTest, BoardLevelStaticTopoAndRanktable)
{
    std::string err;
    ASSERT_TRUE(TopoUtEnv::Instance().Init(err)) << err;
    // 1个超节点 × 16个POD × 8个board = 128个叶子节点, 每board 8卡, 共1024卡
    fs::path clusterDir = TopoUtEnv::Instance().GetClusterDir("ascend950_cluster_16_server_1dpod64", err);
    ASSERT_FALSE(clusterDir.empty()) << err;

    AscendClusterTopoParser::GetInstance().SetClusterStatus(HvmClusterStatus::COMM_DOMAIN_UNINIT);
    fs::path workDir
        = fs::temp_directory_path() / ("hccl_vm_topo_ut_pod_" + std::to_string(static_cast<long>(getpid())));
    std::error_code ec;
    fs::remove_all(workDir, ec);
    fs::create_directories(workDir, ec);
    ASSERT_TRUE(ResetSimulatorStorage(workDir)) << "reset simulator storage failed";

    ASSERT_EQ(
        AscendClusterTopoParser::GetInstance().InitClusterTopo(clusterDir.string()), HcclVmResult::HCCL_SIM_SUCCESS)
        << "InitClusterTopo failed for ascend950_cluster_16_server_1dpod64";

    // (1) Server表: 每board一行, board_id取值0~7, (pod_id, server_id, board_id)唯一
    auto serversResult
        = sim::runtime::Db::GetByPred<sim::runtime::Server>(HcclSim::Storage::All<sim::runtime::Server>());
    auto servers = serversResult.value_or(std::vector<sim::runtime::Server>{});
    ASSERT_EQ(servers.size(), 128u) << "expect one Server row per board (16 POD x 8 board)";
    std::set<std::tuple<uint64_t, uint32_t, uint32_t>> uniqueKeys;
    std::map<uint64_t, uint32_t> boardIdByServerKey;
    for (const auto& srv : servers) {
        uniqueKeys.emplace(srv.pod_id, srv.server_id, srv.board_id);
        boardIdByServerKey[srv.id] = srv.board_id;
        EXPECT_EQ(srv.pod_id, 0u);
        EXPECT_LT(srv.server_id, 16u);
        EXPECT_LT(srv.board_id, 8u);
    }
    EXPECT_EQ(uniqueKeys.size(), 128u) << "(pod_id, server_id, board_id) must be unique";

    // (2) Host表: 每board一行, host IP的CC段 == board_id, AA/BB == superpod+1/POD+1
    auto hostsResult = sim::runtime::Db::GetByPred<sim::runtime::Host>(HcclSim::Storage::All<sim::runtime::Host>());
    auto hosts = hostsResult.value_or(std::vector<sim::runtime::Host>{});
    ASSERT_EQ(hosts.size(), 128u) << "expect one Host row per board";
    // serverKey → server_id 映射, 用于校验BB段
    std::map<uint64_t, uint32_t> serverIdByKey;
    for (const auto& srv : servers) {
        serverIdByKey[srv.id] = srv.server_id;
    }
    for (const auto& host : hosts) {
        const std::string ip(host.ip_addr);
        const auto boardIt = boardIdByServerKey.find(host.server_id);
        ASSERT_TRUE(boardIt != boardIdByServerKey.end()) << "host without server row, ip=" << ip;
        EXPECT_EQ(IpOctet(ip, 0), 1) << "AA(superpod段) should be sp_id+1, ip=" << ip;
        EXPECT_EQ(IpOctet(ip, 1), static_cast<int>(serverIdByKey[host.server_id]) + 1)
            << "BB(server/POD段) should be srv_id+1, ip=" << ip;
        EXPECT_EQ(IpOctet(ip, 2), static_cast<int>(boardIt->second)) << "CC(board段) should be board_id, ip=" << ip;
        EXPECT_EQ(IpOctet(ip, 3), 0) << "host IP should be AA.BB.CC.0, ip=" << ip;
    }

    // (3) Device表: physical_id为board内物理id[0,7], 每board 8卡
    auto devicesResult
        = sim::runtime::Db::GetByPred<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>());
    auto devices = devicesResult.value_or(std::vector<sim::runtime::Device>{});
    ASSERT_EQ(devices.size(), 1024u) << "expect 128 board x 8 device";
    std::map<uint64_t, std::set<uint32_t>> phyIdsByServerKey;
    for (const auto& dev : devices) {
        EXPECT_LT(dev.physical_id, 8u) << "physical_id must be board-local ([0,7]), got " << dev.physical_id;
        phyIdsByServerKey[dev.server_id].insert(dev.physical_id);
    }
    ASSERT_EQ(phyIdsByServerKey.size(), 128u);
    for (const auto& [serverKey, phyIds] : phyIdsByServerKey) {
        EXPECT_EQ(phyIds.size(), 8u) << "each board should hold 8 distinct physical ids";
        EXPECT_EQ(*phyIds.begin(), 0u);
        EXPECT_EQ(*phyIds.rbegin(), 7u);
    }

    // (4) mock-comm(11_64: 1个POD × 8 board × 8卡 = 64 rank)后:
    //     used_dev_num == 8(业务侧aclrtGetDeviceCount()返回值), ranktable的device_id/local_id ∈ [0,7],
    //     layer0的net_instance_id每board唯一
    TopoMeta topoMeta;
    ASSERT_TRUE(ParseYamlTopo("11_64", topoMeta)) << "ParseYamlTopo failed for 11_64";
    ASSERT_EQ(
        AscendClusterTopoParser::GetInstance().InitCommunicationDomain(topoMeta, false), HcclVmResult::HCCL_SIM_SUCCESS)
        << "InitCommunicationDomain failed for 11_64";

    for (const auto& srv : servers) {
        if (srv.pod_id != 0 || srv.server_id != 0) {
            continue; // 11_64只选中 superpod0/server0 的8个board
        }
        auto refreshed = sim::runtime::Db::GetById<sim::runtime::Server>(srv.id);
        ASSERT_TRUE(refreshed.ok());
        EXPECT_EQ(refreshed->used_dev_num, 8u)
            << "used_dev_num should be per-board device count(8), board_id=" << srv.board_id;
    }

    fs::path ranktable = fs::path(InstallPath::ResolveToInstallRoot("data/ranktable.json"));
    ASSERT_TRUE(fs::exists(ranktable)) << "ranktable.json not generated";
    std::ifstream ifs(ranktable);
    json rt;
    ifs >> rt;
    ASSERT_EQ(rt["rank_count"].get<uint32_t>(), 64u);
    std::map<std::string, int> layer0InstanceCount;
    std::set<int> globalLocalIds;
    static const std::regex boardSegRe("_board_(\\d+)_");
    for (const auto& rank : rt["rank_list"]) {
        EXPECT_LT(rank["device_id"].get<uint32_t>(), 8u) << "device_id must be board-local physical id";
        // local_id为POD全局id = board_id*8 + board内物理id(与POD级topo.json的local_id值域一致);
        // board_id从layer0的net_instance_id中board_{b}段解析
        int boardFromNiid = -1;
        for (const auto& level : rank["level_list"]) {
            if (level["net_layer"].get<int>() != 0) {
                continue;
            }
            const std::string niid = level["net_instance_id"].get<std::string>();
            std::smatch m;
            if (std::regex_search(niid, m, boardSegRe)) {
                boardFromNiid = std::stoi(m[1].str());
            }
            layer0InstanceCount[niid]++;
        }
        ASSERT_GE(boardFromNiid, 0) << "layer0 net_instance_id should carry board segment";
        EXPECT_EQ(rank["local_id"].get<int>(), boardFromNiid * 8 + rank["device_id"].get<int>())
            << "local_id must be pod-global: board_id*8 + device_id";
        globalLocalIds.insert(rank["local_id"].get<int>());
    }
    // 64个rank的POD全局local_id互不重复, 覆盖[0,63]
    EXPECT_EQ(globalLocalIds.size(), 64u) << "pod-global local_id must be unique across boards";
    EXPECT_EQ(*globalLocalIds.begin(), 0);
    EXPECT_EQ(*globalLocalIds.rbegin(), 63);
    // 8个board各一个layer0网络实例, 每实例8个rank; 实例名必须携带board_{b}段
    ASSERT_EQ(layer0InstanceCount.size(), 8u) << "layer0 net_instance_id must be unique per board";
    for (const auto& [niid, cnt] : layer0InstanceCount) {
        EXPECT_EQ(cnt, 8) << "each board-level layer0 instance should hold 8 ranks, niid=" << niid;
        EXPECT_NE(niid.find("_board_"), std::string::npos)
            << "layer0 net_instance_id should carry board segment: " << niid;
    }
}
