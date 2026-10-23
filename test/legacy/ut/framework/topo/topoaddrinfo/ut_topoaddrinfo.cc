/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <thread>
#include <time.h>
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <cstdlib>
#include <string>
#include <stdint.h>
#include <ctype.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <vector>
#include <pthread.h>
#include "securec.h"
#include "topo_addr_info.h"
#include "topo_addr_info_log.h"
#include "rank_info_types.h"
#include "hal.h"
#include "product_server.h"

extern "C" int load_dcmi();
extern "C" void reinit();
extern "C" const UBEntity* GetUBEntityByFilter(const UEList* ueList, int dieId, int ueId, int type);

/**
 * @brief 将32字符十六进制字符串转为16字节二进制数组
 * @param hex_str  输入：32位十六进制字符串（必须以'\0'结尾）
 * @param bin_out  输出：16字节uint8_t数组
 * @return 成功0，失败-1（非法字符）
 */
int hex32_to_bin16(const char* hex_str, uint8_t* bin_out)
{
    for (int i = 0; i < 16; i++) {
        // 取两个十六进制字符
        char c1 = hex_str[2 * i];
        char c2 = hex_str[2 * i + 1];

        // 转数字（0-15）
        int h1 = tolower((unsigned char)c1);
        int h2 = tolower((unsigned char)c2);

        h1 = (h1 >= '0' && h1 <= '9') ? h1 - '0' : (h1 >= 'a' && h1 <= 'f') ? 10 + h1 - 'a' : -1;

        h2 = (h2 >= '0' && h2 <= '9') ? h2 - '0' : (h2 >= 'a' && h2 <= 'f') ? 10 + h2 - 'a' : -1;

        // 非法字符检查
        if (h1 < 0 || h2 < 0)
            return -1;

        // 组合成1字节：高4位 + 低4位
        bin_out[i] = (h1 << 4) | h2;
    }
    return 0;
}

/* UT 日志回调：截获 TOPO_* 宏输出到全局缓冲，供用例做关键字断言。
 * 因多线程用例会并发触发日志，对 g_utLogLines 加锁保证线程安全。 */
static std::vector<std::string> g_utLogLines;
static pthread_mutex_t g_utLogMutex = PTHREAD_MUTEX_INITIALIZER;

static void UtLogRecord(int moduleId, int level, const char* fmt, ...)
{
    char buf[1024] = {0};
    va_list args;
    va_start(args, fmt);
    (void)vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    printf("[UT_TOPO] %s\n", buf);
    pthread_mutex_lock(&g_utLogMutex);
    g_utLogLines.push_back(buf);
    pthread_mutex_unlock(&g_utLogMutex);
}

static int UtCheckLogLevel(int moduleId, int logLevel) { return 1; }

static void UtLogReset()
{
    pthread_mutex_lock(&g_utLogMutex);
    g_utLogLines.clear();
    pthread_mutex_unlock(&g_utLogMutex);
}

static bool UtLogContains(const std::string& keyword)
{
    bool found = false;
    pthread_mutex_lock(&g_utLogMutex);
    for (const auto& line : g_utLogLines) {
        if (line.find(keyword) != std::string::npos) {
            found = true;
            break;
        }
    }
    pthread_mutex_unlock(&g_utLogMutex);
    return found;
}

/* 文件操作 mock：拦截 fopen/fstat/fread/fclose，隔离 /etc/hccl_rootinfo.json。
 * 默认让该路径 fopen 返回 NULL，使 PassThrough 失败，避免真实文件干扰用例。 */
static struct {
    int passthrough_enable;      /* 1: 对 /etc/hccl_rootinfo.json 返回 tmpfile，启用 PassThrough 路径 */
    int fstat_fail;              /* 1: fstat 返回 -1 */
    size_t fread_short_read;     /* >0: fread 仅返回该字节数（一次性） */
    const char* tmpfile_content; /* passthrough 时写入 tmpfile 的内容 */
    FILE* mock_fp;               /* 最近一次 mock 出的 tmpfile */
    int mock_fclose_cnt;         /* mock_fp 被 fclose 的次数 */
} g_fileMock = {0, 0, 0, NULL, NULL, 0};

/* __real_* 由链接器 --wrap 选项解析，此处仅提供声明供编译器类型检查 */
extern "C" FILE* __real_fopen(const char* path, const char* mode);
extern "C" int __real_fstat(int fd, struct stat* buf);
extern "C" size_t __real_fread(void* ptr, size_t size, size_t nmemb, FILE* stream);
extern "C" int __real_stat(const char* path, struct stat* buf);
extern "C" int __real_fclose(FILE* stream);

/* 只统计 mock 出的 tmpfile：其余 fclose（其他库/用例自身）不计数，避免无关调用干扰断言 */
extern "C" int __wrap_fclose(FILE* stream)
{
    if (stream != NULL && stream == g_fileMock.mock_fp) {
        g_fileMock.mock_fclose_cnt++;
    }
    return __real_fclose(stream);
}

/* 对 /etc/hccl_rootinfo.json 默认 stat 失败，使 TopoAddrInfoGetSize 走 mainboard_id 路径 */
extern "C" int __wrap_stat(const char* path, struct stat* buf)
{
    if (path != NULL && strcmp(path, "/etc/hccl_rootinfo.json") == 0) {
        errno = ENOENT;
        return -1;
    }
    return __real_stat(path, buf);
}

/* 拦截 /etc/hccl_rootinfo.json：默认返回 NULL 让 PassThrough 失败；
 * passthrough_enable 时返回 tmpfile，由调用方 fclose，不持有引用避免 double free */
extern "C" FILE* __wrap_fopen(const char* path, const char* mode)
{
    if (path != NULL && strcmp(path, "/etc/hccl_rootinfo.json") == 0) {
        if (!g_fileMock.passthrough_enable) {
            return NULL;
        }
        FILE* fp = tmpfile();
        if (fp == NULL) {
            return NULL;
        }
        g_fileMock.mock_fp = fp;
        g_fileMock.mock_fclose_cnt = 0;
        if (g_fileMock.tmpfile_content != NULL) {
            size_t n = strlen(g_fileMock.tmpfile_content);
            fwrite(g_fileMock.tmpfile_content, 1, n, fp);
            rewind(fp);
        }
        return fp;
    }
    return __real_fopen(path, mode);
}

extern "C" int __wrap_fstat(int fd, struct stat* buf)
{
    if (g_fileMock.fstat_fail) {
        errno = EIO;
        return -1;
    }
    return __real_fstat(fd, buf);
}

extern "C" size_t __wrap_fread(void* ptr, size_t size, size_t nmemb, FILE* stream)
{
    if (g_fileMock.fread_short_read > 0) {
        size_t ret = g_fileMock.fread_short_read;
        g_fileMock.fread_short_read = 0; /* 一次性触发 */
        return ret;
    }
    return __real_fread(ptr, size, nmemb, stream);
}

static void FileMockReset()
{
    g_fileMock.passthrough_enable = 0;
    g_fileMock.fstat_fail = 0;
    g_fileMock.fread_short_read = 0;
    g_fileMock.tmpfile_content = NULL;
    g_fileMock.mock_fp = NULL;
    g_fileMock.mock_fclose_cnt = 0;
}

class TopoAddrInfoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TopoAddrInfo tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "TopoAddrInfo tests tear down." << std::endl; }

    virtual void SetUp()
    {
        FileMockReset();
        UtLogReset();
        g_topo_DlogRecord = UtLogRecord;
        g_topo_CheckLogLevel = UtCheckLogLevel;
        std::cout << "A Test case in TopoAddrInfoTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        FileMockReset();
        g_topo_DlogRecord = NULL;
        g_topo_CheckLogLevel = NULL;
        GlobalMockObject::verify();
        std::cout << "A Test case in TopoAddrInfoTest TearDown" << std::endl;
    }
};

TEST_F(TopoAddrInfoTest, Ut_get_mainbaord_id)
{
    unsigned int mainboard_id = 0;
    int ret = hal_get_mainboard_id(0, &mainboard_id);
    // check
    EXPECT_EQ(mainboard_id, 0);
    EXPECT_EQ(ret, -1);
}

TEST_F(TopoAddrInfoTest, Ut_Card_2Px)
{
    unsigned int m = 3;
    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&m)).will(returnValue(0));

    unsigned int xx = 0;
    int ret = hal_get_mainboard_id(0, &xx);
    EXPECT_EQ(xx, 3);
    EXPECT_EQ(ret, 0);
}

/**
 * 验证标卡2P场景
 */
TEST_F(TopoAddrInfoTest, Ut_Card_2P)
{
    // mock data
    unsigned int m = 0x6A;
    char drv_path[256] = "/usr/local/Ascend2";
    dcmi_urma_eid_info_t eidList[MAX_EID_NUM];
    hex32_to_bin16("000000000000000000100000dfdf0020", eidList[0].eid.raw);
    hex32_to_bin16("000000000000000000100000dfdf0028", eidList[1].eid.raw);
    hex32_to_bin16("000000000000000000100000dfdf0030", eidList[2].eid.raw);
    hex32_to_bin16("000000000000000000100000dfdf0051", eidList[3].eid.raw);
    size_t eidNum = 4;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&m)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(hal_get_eid_list_by_phy_id)
        .stubs()
        .with(mockcpp::any(), outBoundP(eidList, eidNum * sizeof(dcmi_urma_eid_info_t)), outBoundP(&eidNum))
        .will(returnValue(0));

    char* buf = (char*)malloc(4096);
    memset(buf, 0x00, 4096);
    size_t bufSize = 4096;
    int x = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(x, 0);
    printf("[%s]\n", buf);
    EXPECT_TRUE(strstr(buf, "dfdf0051") != 0); // 2P使用portgroup通信，验证portgroup的EID在地址信息中
    free(buf);
}

TEST_F(TopoAddrInfoTest, Ut_Card_2P_GetSize)
{
    // mock data
    unsigned int m = 0x6A;
    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&m)).will(returnValue(0));
    size_t bufSize = 0;
    int ret = TopoAddrInfoGetSize(0, &bufSize);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(bufSize, 2048);
}

TEST_F(TopoAddrInfoTest, Ut_Card_4P)
{
    // mock data
    unsigned int mainboard_id = 0x6C;
    char drv_path[256] = "/usr/local/Ascend2";
    dcmi_urma_eid_info_t eidList[MAX_EID_NUM];
    memset_s(eidList, MAX_EID_NUM * sizeof(dcmi_urma_eid_info_t), 0x00, MAX_EID_NUM * sizeof(dcmi_urma_eid_info_t));

    hex32_to_bin16("000000000000000000100000dfdf0020", eidList[0].eid.raw);
    hex32_to_bin16("000000000000000000100000dfdf0028", eidList[1].eid.raw);
    hex32_to_bin16("000000000000000000100000dfdf0030", eidList[2].eid.raw);
    hex32_to_bin16("000000000000000000100000dfdf0051", eidList[3].eid.raw);
    size_t eidNum = 4;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboard_id)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(hal_get_eid_list_by_phy_id)
        .stubs()
        .with(mockcpp::any(), outBoundP(eidList, eidNum * sizeof(dcmi_urma_eid_info_t)), outBoundP(&eidNum))
        .will(returnValue(0));

    size_t bufSize = 0;
    EXPECT_EQ(TopoAddrInfoGetSize(0, &bufSize), 0);
    char* buf = (char*)malloc(bufSize);
    memset(buf, 0x00, bufSize);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(ret, 0);
    printf("[%s]\n", buf);
    // 4P使用直连口
    EXPECT_TRUE(strstr(buf, "dfdf0051") == NULL);
    free(buf);
}

TEST_F(TopoAddrInfoTest, ut_get_pod_rootinfo_size)
{
    unsigned int mainboardId = 0x07;
    const size_t exceptedSize = 2048;
    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboardId)).will(returnValue(0));
    size_t size = 0;
    int ret = TopoAddrInfoGetSize(0, &size);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(size, exceptedSize);
}

TEST_F(TopoAddrInfoTest, ut_get_unknow_rootinfo_size)
{
    unsigned int mainboardId = 0x100;
    const size_t exceptedSize = 4096;
    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboardId)).will(returnValue(0));
    size_t size = 0;
    int ret = TopoAddrInfoGetSize(0, &size);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(size, exceptedSize);
}

/**
 * 打桩Atlas 950 SuperPoD
 */
void mock_spod_info_for_pod(UEList& ueList)
{
    unsigned int mainboard_id = 0x07;
    char drv_path[256] = "/usr/local/Ascend2";
    struct dcmi_spod_info spinfo;
    spinfo.sdid = 0x00000000;
    spinfo.super_pod_size = 128;
    spinfo.super_pod_id = 1;
    spinfo.server_index = 1; // server_index对应local id为8
    spinfo.chassis_id = 0x00000000;
    spinfo.super_pod_type = 0x00000000;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboard_id)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
    MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));
}

/**
 * 构造一个PoD中的NPU的地址信息， 并检查是否正确
 */
TEST_F(TopoAddrInfoTest, ut_rootinfo_for_pod)
{
    // mock start
    UEList ueList;
    memset_s(&ueList, sizeof(UEList), 0x00, sizeof(UEList));
    hex32_to_bin16("000000000001020000100000df100200", ueList.ueList[0].eidList[0].eid.raw);
    hex32_to_bin16("000000000002020000100000df100300", ueList.ueList[0].eidList[1].eid.raw);
    hex32_to_bin16("00000000003f020000100000df100b00", ueList.ueList[0].eidList[2].eid.raw);
    ueList.ueList[0].eidNum = 3;

    hex32_to_bin16("000000000000030000100000df100100", ueList.ueList[1].eidList[0].eid.raw);
    hex32_to_bin16("00000000003f030000100000df100c00", ueList.ueList[1].eidList[1].eid.raw);
    hex32_to_bin16("000000000008030000100000df100900", ueList.ueList[1].eidList[2].eid.raw);
    hex32_to_bin16("000000000007030000100000df100800", ueList.ueList[1].eidList[3].eid.raw);
    hex32_to_bin16("000000000006030000100000df100700", ueList.ueList[1].eidList[4].eid.raw);
    hex32_to_bin16("000000000005030000100000df100600", ueList.ueList[1].eidList[5].eid.raw);
    hex32_to_bin16("000000000004030000100000df100500", ueList.ueList[1].eidList[6].eid.raw);
    hex32_to_bin16("000000000003030000100000df100400", ueList.ueList[1].eidList[7].eid.raw);
    ueList.ueList[1].eidNum = 8;

    hex32_to_bin16("000000000040020000100000df101100", ueList.ueList[2].eidList[0].eid.raw);
    hex32_to_bin16("00000000007f020000100000df101b00", ueList.ueList[2].eidList[1].eid.raw);
    hex32_to_bin16("000000000046020000100000df101700", ueList.ueList[2].eidList[2].eid.raw);
    hex32_to_bin16("000000000045020000100000df101600", ueList.ueList[2].eidList[3].eid.raw);
    hex32_to_bin16("000000000043020000100000df101400", ueList.ueList[2].eidList[4].eid.raw);
    hex32_to_bin16("000000000042020000100000df101300", ueList.ueList[2].eidList[5].eid.raw);
    hex32_to_bin16("000000000041020000100000df101200", ueList.ueList[2].eidList[6].eid.raw);
    ueList.ueList[2].eidNum = 7;

    hex32_to_bin16("000000000044030000100000df101500", ueList.ueList[3].eidList[0].eid.raw);
    hex32_to_bin16("000000000047030000100000df101800", ueList.ueList[3].eidList[1].eid.raw);
    hex32_to_bin16("00000000007f030000100000df101c00", ueList.ueList[3].eidList[2].eid.raw);
    ueList.ueList[3].eidNum = 3;
    ueList.ueNum = 4;
    mock_spod_info_for_pod(ueList);
    // mock end
    size_t bufSize = 0;
    EXPECT_EQ(TopoAddrInfoGetSize(0, &bufSize), 0);
    char* buf = (char*)malloc(bufSize);
    memset_s(buf, bufSize, 0x00, bufSize);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(ret, 0);
    printf("[%s]\n", buf);

    EXPECT_TRUE(strstr(buf, "\"local_id\": 8") != NULL);
    // 校验PG口EID在地址信息中
    EXPECT_TRUE(strstr(buf, "000000000000030000100000df100100") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000003030000100000df100400") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000004030000100000df100500") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000005030000100000df100600") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000006030000100000df100700") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000007030000100000df100800") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000008030000100000df100900") != NULL);
    // 校验mesh层net type正确
    EXPECT_TRUE(strstr(buf, "TOPO_FILE_DESC") != NULL);
    // 校验clos层net type正确
    EXPECT_TRUE(strstr(buf, "CLOS") != NULL);
    free(buf);
}

/**
 * 构造pod机型添加UB_RTP Entity，验证在UB_RTP影响下，能够选择到正确的mesh UB Entity
 */
TEST_F(TopoAddrInfoTest, ut_rootinfo_for_pod_ub_rtp)
{
    // mock start
    UEList ueList;
    memset_s(&ueList, sizeof(UEList), 0x00, sizeof(UEList));
    // 添加一个UB_RTP EID
    hex32_to_bin16("0000000000080a8000100000df001101", ueList.ueList[0].eidList[0].eid.raw);
    ueList.ueList[0].eidNum = 1;

    // 添加一个mesh的UB Entity
    hex32_to_bin16("000000000000030000100000df100100", ueList.ueList[1].eidList[0].eid.raw);
    hex32_to_bin16("00000000003f030000100000df100c00", ueList.ueList[1].eidList[1].eid.raw);
    hex32_to_bin16("000000000008030000100000df100900", ueList.ueList[1].eidList[2].eid.raw);
    hex32_to_bin16("000000000007030000100000df100800", ueList.ueList[1].eidList[3].eid.raw);
    hex32_to_bin16("000000000006030000100000df100700", ueList.ueList[1].eidList[4].eid.raw);
    hex32_to_bin16("000000000005030000100000df100600", ueList.ueList[1].eidList[5].eid.raw);
    hex32_to_bin16("000000000004030000100000df100500", ueList.ueList[1].eidList[6].eid.raw);
    hex32_to_bin16("000000000003030000100000df100400", ueList.ueList[1].eidList[7].eid.raw);
    ueList.ueList[1].eidNum = 8;
    ueList.ueNum = 2;
    mock_spod_info_for_pod(ueList);
    // mock end
    size_t bufSize = 0;
    EXPECT_EQ(TopoAddrInfoGetSize(0, &bufSize), 0);
    char* buf = (char*)malloc(bufSize);
    memset_s(buf, bufSize, 0x00, bufSize);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(ret, 0);
    printf("[%s]\n", buf);

    EXPECT_TRUE(strstr(buf, "\"local_id\": 8") != NULL);
    // 校验PG口EID在地址信息中
    EXPECT_TRUE(strstr(buf, "000000000000030000100000df100100") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000003030000100000df100400") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000004030000100000df100500") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000005030000100000df100600") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000006030000100000df100700") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000007030000100000df100800") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000008030000100000df100900") != NULL);
    // 校验mesh层net type正确
    EXPECT_TRUE(strstr(buf, "TOPO_FILE_DESC") != NULL);
    // 校验clos层net type正确
    EXPECT_TRUE(strstr(buf, "CLOS") != NULL);
    // 校验UB_RTP未被选中
    EXPECT_TRUE(strstr(buf, "0000000000080a8000100000df001101") == NULL);
    free(buf);
}

int mock_dcmi_init()
{
    // 模拟初始化耗时
    sleep(1);
    return 0;
}

int mock_dcmiv2_get_mainboard_id(int npu_id, unsigned int* mainboard_id)
{
    *mainboard_id = 0x07;
    return 0;
}

// 预期整个生命周期只会被调用1次
int mock_get_logicid_from_chipphy_id(unsigned int phyId, unsigned int* logicId)
{
    *logicId = phyId;
    return 0;
}

int mock_aclrtGetLogicDevIdByPhyDevId(const int phyId, int* const logicId)
{
    *logicId = phyId;
    return 0;
}

int mock_convertId(const int inputId, int* const outputId)
{
    *outputId = inputId;
    return 0;
}

int mock_halGetDeviceInfo(unsigned int devId, uint32_t moduleType, int32_t infoType, int64_t* value)
{
    *value = 0x07;
    return 0;
}

int mock_aclrtGetDeviceCount(uint32_t* count)
{
    *count = 0;
    return 0;
}

void* mock_dlsym(void* handle, const char* symbol)
{
    if (strcmp(symbol, "dcmiv2_init") == 0) {
        return (void*)mock_dcmi_init;
    }
    if (strcmp(symbol, "dcmiv2_get_mainboard_id") == 0) {
        return (void*)mock_dcmiv2_get_mainboard_id;
    }
    if (strcmp(symbol, "dcmiv2_get_dev_id_from_chip_phyid") == 0
        || strcmp(symbol, "dcmiv2_get_dev_id_by_chip_phy_id") == 0) {
        return (void*)mock_get_logicid_from_chipphy_id;
    }

    if (strcmp(symbol, "aclrtGetLogicDevIdByPhyDevId") == 0) {
        return (void*)mock_aclrtGetLogicDevIdByPhyDevId;
    }

    if (strcmp(symbol, "aclrtGetUserDevIdByPhyDevId") == 0) {
        return (void*)mock_convertId;
    }

    if (strcmp(symbol, "aclrtGetLogicDevIdByUserDevId") == 0) {
        return (void*)mock_convertId;
    }

    if (strcmp(symbol, "aclrtGetPhyDevIdByUserDevId") == 0) {
        return (void*)mock_convertId;
    }

    if (strcmp(symbol, "aclrtGetDeviceCount") == 0) {
        return (void*)mock_aclrtGetDeviceCount;
    }

    if (strcmp(symbol, "halGetDeviceInfo") == 0) {
        return (void*)mock_halGetDeviceInfo;
    }
    return (void*)0x1;
}

void* mock_dlopen(const char* filename, int flag) { return (void*)0x1; }

TEST_F(TopoAddrInfoTest, ut_multi_init)
{
    // mock data
    unsigned int mainBoardId1 = 0;
    unsigned int mainBoardId2 = 0;
    unsigned int expectedMainboardId = 0x07;
    MOCKER(hal_dlopen).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(mock_dlopen));
    MOCKER(hal_dlsym).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(mock_dlsym));
    hal_get_mainboard_id(0, &mainBoardId1);
    EXPECT_EQ(mainBoardId1, expectedMainboardId);
    // 连续初始化两次，模拟多线程初始化，第二次进入等待状态
    hal_get_mainboard_id(0, &mainBoardId2);
    EXPECT_EQ(mainBoardId2, expectedMainboardId);
}

/**
 * 验证多线程并发调用
 */
TEST_F(TopoAddrInfoTest, ut_multi_thread_init)
{
    // mock data
    unsigned int expectedMainboardId = 0x07;
    MOCKER(hal_dlopen).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(mock_dlopen));
    MOCKER(hal_dlsym).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(mock_dlsym));
    std::vector<std::thread> ts;
    constexpr int testThreadNum = 16;
    std::vector<unsigned int> mids(testThreadNum, 0);
    std::vector<int> rets(testThreadNum, 0);
    for (int i = 0; i < testThreadNum; i++) {
        ts.emplace_back([i, &mids, &rets] {
            rets[i] = hal_get_mainboard_id(0, &mids[i]);
        });
    }
    for (int i = 0; i < testThreadNum; i++) {
        ts[i].join();
        EXPECT_EQ(rets[i], 0);
        EXPECT_EQ(mids[i], expectedMainboardId);
    }
}

void mock_uelist_for_server_with_uboe(UEList* ueList)
{
    memset_s(ueList, sizeof(UEList), 0x00, sizeof(UEList));
    hex32_to_bin16("000000000000020000100000df000101", ueList->ueList[0].eidList[0].eid.raw);
    ueList->ueList[0].eidNum = 1;
    hex32_to_bin16("000000000001030000100000df000201", ueList->ueList[1].eidList[0].eid.raw);
    hex32_to_bin16("00000000003f030000100000df000b01", ueList->ueList[1].eidList[1].eid.raw);
    hex32_to_bin16("000000000008030000100000df000901", ueList->ueList[1].eidList[2].eid.raw);
    hex32_to_bin16("000000000007030000100000df000801", ueList->ueList[1].eidList[3].eid.raw);
    hex32_to_bin16("000000000006030000100000df000701", ueList->ueList[1].eidList[4].eid.raw);
    hex32_to_bin16("000000000005030000100000df000601", ueList->ueList[1].eidList[5].eid.raw);
    hex32_to_bin16("000000000004030000100000df000501", ueList->ueList[1].eidList[6].eid.raw);
    hex32_to_bin16("000000000003030000100000df000401", ueList->ueList[1].eidList[7].eid.raw);
    hex32_to_bin16("000000000002030000100000df000301", ueList->ueList[1].eidList[8].eid.raw);
    ueList->ueList[1].eidNum = 9;
    hex32_to_bin16("000000000040020000100000df001101", ueList->ueList[2].eidList[0].eid.raw);
    ueList->ueList[2].eidNum = 1;
    hex32_to_bin16("000000000041050000100000df001200", ueList->ueList[3].eidList[0].eid.raw);
    hex32_to_bin16("00000000007f050000100000df001b00", ueList->ueList[3].eidList[1].eid.raw);
    hex32_to_bin16("000000000047050000100000df001800", ueList->ueList[3].eidList[2].eid.raw);
    hex32_to_bin16("000000000046050000100000df001700", ueList->ueList[3].eidList[3].eid.raw);
    hex32_to_bin16("000000000045050000100000df001600", ueList->ueList[3].eidList[4].eid.raw);
    hex32_to_bin16("000000000044050000100000df001500", ueList->ueList[3].eidList[5].eid.raw);
    hex32_to_bin16("000000000043050000100000df001400", ueList->ueList[3].eidList[6].eid.raw);
    hex32_to_bin16("000000000042050000100000df001300", ueList->ueList[3].eidList[7].eid.raw);
    ueList->ueList[3].eidNum = 8;
    hex32_to_bin16("0000000000ff0ac0000000000a140200", ueList->ueList[4].eidList[0].eid.raw); // 这一条是UBOE的EID
    hex32_to_bin16("0000000000fffec0000000006a610c00", ueList->ueList[4].eidList[1].eid.raw);
    ueList->ueList[4].eidNum = 2;
    ueList->ueNum = 5;
}

void mock_uelist_for_server_without_uboe(UEList* ueList)
{
    memset_s(ueList, sizeof(UEList), 0x00, sizeof(UEList));
    hex32_to_bin16("000000000000020000100000df000101", ueList->ueList[0].eidList[0].eid.raw);
    ueList->ueList[0].eidNum = 1;
    hex32_to_bin16("000000000001030000100000df000201", ueList->ueList[1].eidList[0].eid.raw);
    hex32_to_bin16("00000000003f030000100000df000b01", ueList->ueList[1].eidList[1].eid.raw);
    hex32_to_bin16("000000000008030000100000df000901", ueList->ueList[1].eidList[2].eid.raw);
    hex32_to_bin16("000000000007030000100000df000801", ueList->ueList[1].eidList[3].eid.raw);
    hex32_to_bin16("000000000006030000100000df000701", ueList->ueList[1].eidList[4].eid.raw);
    hex32_to_bin16("000000000005030000100000df000601", ueList->ueList[1].eidList[5].eid.raw);
    hex32_to_bin16("000000000004030000100000df000501", ueList->ueList[1].eidList[6].eid.raw);
    hex32_to_bin16("000000000003030000100000df000401", ueList->ueList[1].eidList[7].eid.raw);
    hex32_to_bin16("000000000002030000100000df000301", ueList->ueList[1].eidList[8].eid.raw);
    ueList->ueList[1].eidNum = 9;
    hex32_to_bin16("000000000040020000100000df001101", ueList->ueList[2].eidList[0].eid.raw);
    ueList->ueList[2].eidNum = 1;
    hex32_to_bin16("000000000041030000100000df001200", ueList->ueList[3].eidList[0].eid.raw);
    hex32_to_bin16("00000000007f030000100000df001b00", ueList->ueList[3].eidList[1].eid.raw);
    hex32_to_bin16("000000000047030000100000df001800", ueList->ueList[3].eidList[2].eid.raw);
    hex32_to_bin16("000000000046030000100000df001700", ueList->ueList[3].eidList[3].eid.raw);
    hex32_to_bin16("000000000045030000100000df001600", ueList->ueList[3].eidList[4].eid.raw);
    hex32_to_bin16("000000000044030000100000df001500", ueList->ueList[3].eidList[5].eid.raw);
    hex32_to_bin16("000000000043030000100000df001400", ueList->ueList[3].eidList[6].eid.raw);
    hex32_to_bin16("000000000042030000100000df001300", ueList->ueList[3].eidList[7].eid.raw);
    ueList->ueList[3].eidNum = 8;
    ueList->ueNum = 4;
}

TEST_F(TopoAddrInfoTest, ut_rootinfo_for_server_no_uboe1)
{
    /*
    模拟无UBOE机型， 但是输入有UBOE机型的配置，且已配置UBOE IP地址
    输入UBOE机型EID，但该机型没有UBOE, 因此即使输入了UBOE的UE会被过滤掉
    预期正常输出，但是不带UBOE IP地址
    */
    unsigned int mainboard_id = 0x25; // 这种机型没有UBOE
    char drv_path[256] = "/usr/local/Ascend2";
    UEList ueList;
    mock_uelist_for_server_with_uboe(&ueList);

    struct dcmi_spod_info spinfo;
    spinfo.sdid = 0x00000000;
    spinfo.super_pod_size = 128;
    spinfo.super_pod_id = 1;
    spinfo.server_index = 1;
    spinfo.chassis_id = 0x00000000;
    spinfo.super_pod_type = 0x00000000;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboard_id)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
    MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));

    size_t bufSize = 0;
    EXPECT_EQ(TopoAddrInfoGetSize(0, &bufSize), 0);
    char* buf = (char*)malloc(bufSize);
    memset_s(buf, bufSize, 0x00, bufSize);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(ret, 0);
    printf("[%s]\n", buf);

    // 校验MESH
    EXPECT_TRUE(strstr(buf, "000000000041050000100000df001200") != NULL);
    EXPECT_TRUE(strstr(buf, "00000000007f050000100000df001b00") == NULL); // mesh中的PG不在其中
    EXPECT_TRUE(strstr(buf, "000000000047050000100000df001800") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000046050000100000df001700") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000045050000100000df001600") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000044050000100000df001500") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000043050000100000df001400") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000042050000100000df001300") != NULL);

    // 校验UBOE IP, 不应该存在
    EXPECT_TRUE(strstr(buf, "10.10.20.2") == NULL); // 虽然有UBOE IP，但是mainboard机型不带UBOE

    // 校验clos端口必须都在
    for (int i = 1; i <= 8; i++) {
        char port[32] = {0};
        sprintf_s(port, sizeof(port), "0/%d", i);
        EXPECT_TRUE(strstr(buf, port) != NULL);
    }

    // 校验mesh层net type正确
    EXPECT_TRUE(strstr(buf, "TOPO_FILE_DESC") != NULL);
    // 校验clos层net type正确
    free(buf);
}

TEST_F(TopoAddrInfoTest, ut_rootinfo_for_server_no_uboe2)
{
    /*
    模拟无UBOE机型， 但是输入无UBOE机型的urma配置，未配置UBOE IP地址
    预期正常输出
    */
    unsigned int mainboard_id = 0x27; // 这种机型没有UBOE
    char drv_path[256] = "/usr/local/Ascend2";
    UEList ueList;
    mock_uelist_for_server_without_uboe(&ueList);

    struct dcmi_spod_info spinfo;
    spinfo.sdid = 0x00000000;
    spinfo.super_pod_size = 128;
    spinfo.super_pod_id = 1;
    spinfo.server_index = 1;
    spinfo.chassis_id = 0x00000000;
    spinfo.super_pod_type = 0x00000000;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboard_id)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
    MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));

    size_t bufSize = 0;
    EXPECT_EQ(TopoAddrInfoGetSize(0, &bufSize), 0);
    char* buf = (char*)malloc(bufSize);
    memset_s(buf, bufSize, 0x00, bufSize);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(ret, 0);
    printf("[%s]\n", buf);

    // 校验MESH, 未配置UBOE，FE ID是3
    EXPECT_TRUE(strstr(buf, "000000000041030000100000df001200") != NULL);
    EXPECT_TRUE(strstr(buf, "00000000007f030000100000df001b00") == NULL); // mesh中的PG不在其中
    EXPECT_TRUE(strstr(buf, "000000000047030000100000df001800") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000046030000100000df001700") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000045030000100000df001600") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000044030000100000df001500") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000043030000100000df001400") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000042030000100000df001300") != NULL);

    // 校验clos端口必须都在
    for (int i = 1; i <= 8; i++) {
        char port[32] = {0};
        sprintf_s(port, sizeof(port), "0/%d", i);
        EXPECT_TRUE(strstr(buf, port) != NULL);
    }

    // 校验mesh层net type正确
    EXPECT_TRUE(strstr(buf, "TOPO_FILE_DESC") != NULL);
    // 校验clos层net type正确
    free(buf);
}

TEST_F(TopoAddrInfoTest, ut_rootinfo_for_server_uboe)
{
    /*
    模拟有UBOE机型， 输入有UBOE机型的配置，且已配置UBOE IP地址
    预期正常输出，带UBOE IP地址
     */
    unsigned int mainboard_id = 0x27;
    char drv_path[256] = "/usr/local/Ascend2";
    UEList ueList;
    mock_uelist_for_server_with_uboe(&ueList);

    struct dcmi_spod_info spinfo;
    spinfo.sdid = 0x00000000;
    spinfo.super_pod_size = 128;
    spinfo.super_pod_id = 1;
    spinfo.server_index = 1;
    spinfo.chassis_id = 0x00000000;
    spinfo.super_pod_type = 0x00000000;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboard_id)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
    MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));

    size_t bufSize = 0;
    EXPECT_EQ(TopoAddrInfoGetSize(0, &bufSize), 0);
    char* buf = (char*)malloc(bufSize);
    memset_s(buf, bufSize, 0x00, bufSize);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(ret, 0);
    printf("[%s]\n", buf);
    // 校验UBOE IP
    EXPECT_TRUE(strstr(buf, "10.10.20.2") != NULL);
    free(buf);
}

TEST_F(TopoAddrInfoTest, ut_rootinfo_for_350L)
{
    /*
    打桩350L机型, 预期正常输出, 输出的0层地址中包含一个mesh组网和一个clos组网
     */
    unsigned int mainboard_id = 0x44;
    char drv_path[256] = "/usr/local/Ascend2";
    UEList ueList;
    memset_s(&ueList, sizeof(UEList), 0x00, sizeof(UEList));
    hex32_to_bin16("000000000f44020000100000df00a8f8", ueList.ueList[0].eidList[0].eid.raw);
    hex32_to_bin16("000000000f7f020000100000df00d9f8", ueList.ueList[0].eidList[1].eid.raw);
    hex32_to_bin16("000000000f47020000100000df00c0f8", ueList.ueList[0].eidList[2].eid.raw);
    hex32_to_bin16("000000000f46020000100000df00b8f8", ueList.ueList[0].eidList[3].eid.raw);
    hex32_to_bin16("000000000f45020000100000df00b0f8", ueList.ueList[0].eidList[4].eid.raw);
    ueList.ueList[0].eidNum = 5;

    hex32_to_bin16("000000000f40030000100000df0088f8", ueList.ueList[1].eidList[0].eid.raw);
    hex32_to_bin16("000000000f7f030000100000df00d8f8", ueList.ueList[1].eidList[1].eid.raw);
    hex32_to_bin16("000000000f42030000100000df0098f8", ueList.ueList[1].eidList[2].eid.raw);
    hex32_to_bin16("000000000f41030000100000df0090f8", ueList.ueList[1].eidList[3].eid.raw);
    ueList.ueList[1].eidNum = 4;
    ueList.ueNum = 2;

    struct dcmi_spod_info spinfo;
    spinfo.sdid = 0x00000000;
    spinfo.super_pod_size = 128;
    spinfo.super_pod_id = 1;
    spinfo.server_index = 1;
    spinfo.chassis_id = 0x00000000;
    spinfo.super_pod_type = 0x00000000;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboard_id)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
    MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));

    size_t bufSize = 0;
    EXPECT_EQ(TopoAddrInfoGetSize(0, &bufSize), 0);
    char* buf = (char*)malloc(bufSize);
    memset_s(buf, bufSize, 0x00, bufSize);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(ret, 0);
    printf("[%s]\n", buf);

    // 校验mesh部分
    EXPECT_TRUE(strstr(buf, "000000000f40030000100000df0088f8") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000f42030000100000df0098f8") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000f41030000100000df0090f8") != NULL);
    //  校验CLOS地址
    EXPECT_TRUE(strstr(buf, "000000000f7f020000100000df00d9f8") != NULL);

    // 校验mesh层net type正确
    EXPECT_TRUE(strstr(buf, "TOPO_FILE_DESC") != NULL);
    // 校验clos层net type正确
    free(buf);
}

TEST_F(TopoAddrInfoTest, ut_rootinfo_for_ubx2)
{
    /*
    打桩UBX机型, 预期正常输出, 输出的0层地址中包含一个mesh组网和一个clos组网
     */
    unsigned int mainboard_id = 0x44;
    char drv_path[256] = "/usr/local/Ascend2";
    UEList ueList;
    memset_s(&ueList, sizeof(UEList), 0x00, sizeof(UEList));
    hex32_to_bin16("000000000f44020000100000df00a8f8", ueList.ueList[0].eidList[0].eid.raw);
    hex32_to_bin16("000000000f7f020000100000df00d9f8", ueList.ueList[0].eidList[1].eid.raw);
    hex32_to_bin16("000000000f47020000100000df00c0f8", ueList.ueList[0].eidList[2].eid.raw);
    hex32_to_bin16("000000000f46020000100000df00b8f8", ueList.ueList[0].eidList[3].eid.raw);
    hex32_to_bin16("000000000f45020000100000df00b0f8", ueList.ueList[0].eidList[4].eid.raw);
    ueList.ueList[0].eidNum = 5;

    hex32_to_bin16("000000000f40030000100000df0088f8", ueList.ueList[1].eidList[0].eid.raw);
    hex32_to_bin16("000000000f7f030000100000df00d8f8", ueList.ueList[1].eidList[1].eid.raw);
    hex32_to_bin16("000000000f42030000100000df0098f8", ueList.ueList[1].eidList[2].eid.raw);
    hex32_to_bin16("000000000f41030000100000df0090f8", ueList.ueList[1].eidList[3].eid.raw);
    ueList.ueList[1].eidNum = 4;
    ueList.ueNum = 2;

    struct dcmi_spod_info spinfo;
    spinfo.sdid = 0x00000000;
    spinfo.super_pod_size = 128;
    spinfo.super_pod_id = 1;
    spinfo.server_index = 1;
    spinfo.chassis_id = 0x00000000;
    spinfo.super_pod_type = 4; // UBX的superpod type 为4

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboard_id)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
    MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));

    size_t bufSize = 4096;
    char* buf = (char*)malloc(bufSize);
    memset_s(buf, bufSize, 0x00, bufSize);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(ret, 0);
    printf("[%s]\n", buf);

    // 校验mesh部分
    EXPECT_TRUE(strstr(buf, "000000000f40030000100000df0088f8") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000f42030000100000df0098f8") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000f41030000100000df0090f8") != NULL);
    //  校验CLOS地址
    EXPECT_TRUE(strstr(buf, "000000000f7f020000100000df00d9f8") != NULL);

    // 校验mesh层net type正确
    EXPECT_TRUE(strstr(buf, "TOPO_FILE_DESC") != NULL);
    // 校验clos层net type正确
    free(buf);
}

TEST_F(TopoAddrInfoTest, ut_rootinfo_for_server_uboe_16fm)
{
    /*
    模拟有UBOE机型， 输入有UBOE机型的配置，且已配置UBOE IP地址
    预期正常输出，带UBOE IP地址
     */
    unsigned int mainboard_id = 0x27;
    char drv_path[256] = "/usr/local/Ascend2";
    UEList ueList;
    mock_uelist_for_server_with_uboe(&ueList);

    struct dcmi_spod_info spinfo;
    spinfo.sdid = 0x00000000;
    spinfo.super_pod_size = 128;
    spinfo.super_pod_id = 1;
    spinfo.server_index = 1;
    spinfo.chassis_id = 0x00000000;
    spinfo.super_pod_type = TOPO_TYPE_SERVER_16FM;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboard_id)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
    MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));

    size_t bufSize = 4096;
    size_t memSize = 0;
    EXPECT_EQ(TopoAddrInfoGetSize(0, &memSize), 0);
    char* buf = (char*)malloc(bufSize);
    memset_s(buf, bufSize, 0x00, bufSize);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(bufSize <= memSize) << "bufSize = " << bufSize << " memSize = " << memSize;
    printf("[%s]\n", buf);

    //  跨节点MESH
    EXPECT_TRUE(strstr(buf, "000000000001030000100000df000201") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000008030000100000df000901") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000007030000100000df000801") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000006030000100000df000701") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000005030000100000df000601") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000004030000100000df000501") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000003030000100000df000401") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000002030000100000df000301") != NULL);

    //  本节点MESH
    EXPECT_TRUE(strstr(buf, "000000000041050000100000df001200") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000047050000100000df001800") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000046050000100000df001700") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000045050000100000df001600") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000044050000100000df001500") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000043050000100000df001400") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000042050000100000df001300") != NULL);

    // 校验UBOE IP
    EXPECT_TRUE(strstr(buf, "10.10.20.2") != NULL);

    // 16p fullmesh 的 mesh 层 net instance id 应为超节点级 sp_1
    EXPECT_TRUE(strstr(buf, "\"net_instance_id\": \"sp_1\"") != NULL);
    // 不应再出现服务器级 net instance id sp_1_srv_1
    EXPECT_TRUE(strstr(buf, "\"net_instance_id\": \"sp_1_srv_1\"") == NULL);

    free(buf);
}

void mock_uelist_for_pod_flex(UEList* ueList)
{
    memset_s(ueList, sizeof(UEList), 0x00, sizeof(UEList));
    hex32_to_bin16("000000000000020000100000df000101", ueList->ueList[0].eidList[0].eid.raw);
    ueList->ueList[0].eidNum = 1;
    hex32_to_bin16("0000000000080a8000100000df001101", ueList->ueList[1].eidList[0].eid.raw);
    ueList->ueList[1].eidNum = 1;
    hex32_to_bin16("000000000001040000100000df001200", ueList->ueList[2].eidList[0].eid.raw);
    hex32_to_bin16("00000000003f040000100000df001b00", ueList->ueList[2].eidList[1].eid.raw);
    hex32_to_bin16("000000000007040000100000df001800", ueList->ueList[2].eidList[2].eid.raw);
    hex32_to_bin16("000000000006040000100000df001700", ueList->ueList[2].eidList[3].eid.raw);
    hex32_to_bin16("000000000005040000100000df001600", ueList->ueList[2].eidList[4].eid.raw);
    hex32_to_bin16("000000000004040000100000df001500", ueList->ueList[2].eidList[5].eid.raw);
    hex32_to_bin16("000000000003040000100000df001400", ueList->ueList[2].eidList[6].eid.raw);
    hex32_to_bin16("000000000002040000100000df001300", ueList->ueList[2].eidList[7].eid.raw);
    ueList->ueList[2].eidNum = 8;
    hex32_to_bin16("000000000040020000100000df001200", ueList->ueList[3].eidList[0].eid.raw);
    hex32_to_bin16("00000000007f020000100000df001b00", ueList->ueList[3].eidList[1].eid.raw);
    hex32_to_bin16("000000000047020000100000df001800", ueList->ueList[3].eidList[2].eid.raw);
    hex32_to_bin16("000000000046020000100000df001800", ueList->ueList[3].eidList[3].eid.raw);
    hex32_to_bin16("000000000045020000100000df001700", ueList->ueList[3].eidList[4].eid.raw);
    hex32_to_bin16("000000000044020000100000df001600", ueList->ueList[3].eidList[5].eid.raw);
    hex32_to_bin16("000000000043020000100000df001500", ueList->ueList[3].eidList[6].eid.raw);
    hex32_to_bin16("000000000042020000100000df001400", ueList->ueList[3].eidList[7].eid.raw);
    hex32_to_bin16("000000000041020000100000df001300", ueList->ueList[3].eidList[8].eid.raw);
    ueList->ueList[3].eidNum = 9;
    ueList->ueNum = 4;
}

TEST_F(TopoAddrInfoTest, ut_rootinfo_for_pod_flex)
{
    /*
    打桩Atlas 950 SuperPoD Flex机型
    预期正常输出，带UBG地址
    */
    unsigned int mainboard_id = 0x2f;
    char drv_path[256] = "/usr/local/Ascend2";
    UEList ueList;
    mock_uelist_for_pod_flex(&ueList);

    struct dcmi_spod_info spinfo;
    spinfo.sdid = 0x00000000;
    spinfo.super_pod_size = 128;
    spinfo.super_pod_id = 1;
    spinfo.server_index = 1;
    spinfo.chassis_id = 0x00000000;
    spinfo.super_pod_type = 0;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboard_id)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
    MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));

    size_t memSize = 0;
    EXPECT_EQ(TopoAddrInfoGetSize(0, &memSize), 0);
    size_t bufSize = 4096;
    char* buf = (char*)malloc(bufSize);
    memset_s(buf, bufSize, 0x00, bufSize);
    int ret = TopoAddrInfoGet(9, buf, &bufSize);
    EXPECT_TRUE(bufSize <= memSize);
    EXPECT_EQ(ret, 0);
    printf("[%s]\n", buf);

    EXPECT_TRUE(strstr(buf, "sp_1_srv_1_board1") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000001040000100000df001200") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000007040000100000df001800") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000006040000100000df001700") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000005040000100000df001600") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000004040000100000df001500") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000003040000100000df001400") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000002040000100000df001300") != NULL);

    EXPECT_TRUE(strstr(buf, "00000000007f020000100000df001b00") != NULL);

    // 校验UBG EID
    EXPECT_TRUE(strstr(buf, "0000000000080a8000100000df001101") != NULL);

    free(buf);
}

void mock_uelist_for_pod_server_550_100(UEList* ueList)
{
    memset_s(ueList, sizeof(UEList), 0x00, sizeof(UEList));
    hex32_to_bin16("000000000f40020000100000df07b1f9", ueList->ueList[0].eidList[0].eid.raw);
    hex32_to_bin16("000000000f7f020000100000df07bbf9", ueList->ueList[0].eidList[1].eid.raw);
    hex32_to_bin16("000000000f47020000100000df07b8f9", ueList->ueList[0].eidList[2].eid.raw);
    hex32_to_bin16("000000000f46020000100000df07b7f9", ueList->ueList[0].eidList[3].eid.raw);
    hex32_to_bin16("000000000f45020000100000df07b6f9", ueList->ueList[0].eidList[4].eid.raw);
    hex32_to_bin16("000000000f44020000100000df07b5f9", ueList->ueList[0].eidList[5].eid.raw);
    hex32_to_bin16("000000000f43020000100000df07b4f9", ueList->ueList[0].eidList[6].eid.raw);
    hex32_to_bin16("000000000f42020000100000df07b3f9", ueList->ueList[0].eidList[7].eid.raw);
    hex32_to_bin16("000000000f41020000100000df07b2f9", ueList->ueList[0].eidList[8].eid.raw);
    ueList->ueList[0].eidNum = 9;
    ueList->ueNum = 1;
}

TEST_F(TopoAddrInfoTest, ut_rootinfo_for_serv_550EL_100)
{
    /*
    打桩Atlas 550EL
    */
    unsigned int mainboard_id = MAIN_BOARD_ID_SERVER_550EL_100;
    char drv_path[256] = "/usr/local/Ascend2";
    UEList ueList;
    mock_uelist_for_pod_server_550_100(&ueList);

    struct dcmi_spod_info spinfo;
    spinfo.sdid = 0x00000000;
    spinfo.super_pod_size = 128;
    spinfo.super_pod_id = 1;
    spinfo.server_index = 1;
    spinfo.chassis_id = 0x00000000;
    spinfo.super_pod_type = 0;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboard_id)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
    MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));

    size_t memSize = 0;
    EXPECT_EQ(TopoAddrInfoGetSize(0, &memSize), 0);
    size_t bufSize = 4096;
    char* buf = (char*)malloc(bufSize);
    memset_s(buf, bufSize, 0x00, bufSize);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_TRUE(bufSize <= memSize);
    EXPECT_EQ(ret, 0);
    printf("[%s]\n", buf);
    EXPECT_TRUE(strstr(buf, "000000000f7f020000100000df07bbf9") != NULL);
    free(buf);
}

void mock_uelist_for_pod_server_550_200(UEList* ueList)
{
    memset_s(ueList, sizeof(UEList), 0x00, sizeof(UEList));
    hex32_to_bin16("000000000f04020000100000df07a5f9", ueList->ueList[0].eidList[0].eid.raw);
    hex32_to_bin16("000000000f3f020000100000df07abf9", ueList->ueList[0].eidList[1].eid.raw);
    hex32_to_bin16("000000000f07020000100000df07a8f9", ueList->ueList[0].eidList[2].eid.raw);
    hex32_to_bin16("000000000f06020000100000df07a7f9", ueList->ueList[0].eidList[3].eid.raw);
    hex32_to_bin16("000000000f05020000100000df07a6f9", ueList->ueList[0].eidList[4].eid.raw);
    ueList->ueList[0].eidNum = 5;
    hex32_to_bin16("000000000f40020000100000df07b1f9", ueList->ueList[1].eidList[0].eid.raw);
    hex32_to_bin16("000000000f7f020000100000df07bbf9", ueList->ueList[1].eidList[1].eid.raw);
    hex32_to_bin16("000000000f47020000100000df07b8f9", ueList->ueList[1].eidList[2].eid.raw);
    hex32_to_bin16("000000000f46020000100000df07b7f9", ueList->ueList[1].eidList[3].eid.raw);
    hex32_to_bin16("000000000f45020000100000df07b6f9", ueList->ueList[1].eidList[4].eid.raw);
    hex32_to_bin16("000000000f44020000100000df07b5f9", ueList->ueList[1].eidList[5].eid.raw);
    hex32_to_bin16("000000000f43020000100000df07b4f9", ueList->ueList[1].eidList[6].eid.raw);
    hex32_to_bin16("000000000f42020000100000df07b3f9", ueList->ueList[1].eidList[7].eid.raw);
    hex32_to_bin16("000000000f41020000100000df07b2f9", ueList->ueList[1].eidList[8].eid.raw);
    ueList->ueList[1].eidNum = 9;
    ueList->ueNum = 2;
}

TEST_F(TopoAddrInfoTest, ut_rootinfo_for_serv_550EL_200)
{
    /*
    打桩Atlas 550EL
    */
    unsigned int mainboard_id = MAIN_BOARD_ID_SERVER_550EL_200;
    char drv_path[256] = "/usr/local/Ascend2";
    UEList ueList;
    mock_uelist_for_pod_server_550_200(&ueList);

    struct dcmi_spod_info spinfo;
    spinfo.sdid = 0x00000000;
    spinfo.super_pod_size = 128;
    spinfo.super_pod_id = 1;
    spinfo.server_index = 1;
    spinfo.chassis_id = 0x00000000;
    spinfo.super_pod_type = 0;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboard_id)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
    MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));

    size_t bufSize = 4096;
    char* buf = (char*)malloc(bufSize);
    memset_s(buf, bufSize, 0x00, bufSize);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(ret, 0);
    printf("[%s]\n", buf);
    EXPECT_TRUE(strstr(buf, "000000000f3f020000100000df07abf9") != NULL);
    EXPECT_TRUE(strstr(buf, "000000000f7f020000100000df07bbf9") != NULL);
    free(buf);
}

/**
 * @brief 验证 PassThrough 中 fread 短读时返回错误
 */
TEST_F(TopoAddrInfoTest, Ut_PassThrough_FreadShortRead)
{
    g_fileMock.passthrough_enable = 1;
    g_fileMock.tmpfile_content = "0123456789abcdefghij";
    g_fileMock.fread_short_read = 10; /* 文件 20 字节，fread 仅返回 10，构造短读 */

    /* PassThrough 失败后 TopoAddrInfoGet 回退到 mainboard_id 路径，mock 失败使其快速返回 */
    unsigned int m = 0;
    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&m)).will(returnValue(-1));

    char buf[4096];
    memset(buf, 0, sizeof(buf));
    size_t bufSize = sizeof(buf);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_NE(ret, 0);
    EXPECT_TRUE(UtLogContains("short read")) << "应输出 short read 错误日志";
}

/**
 * @brief 验证 PassThrough 中 fstat 失败时返回错误
 */
TEST_F(TopoAddrInfoTest, Ut_PassThrough_FstatFail)
{
    g_fileMock.passthrough_enable = 1;
    g_fileMock.tmpfile_content = "any_content";
    g_fileMock.fstat_fail = 1;

    /* PassThrough 失败后 TopoAddrInfoGet 回退到 mainboard_id 路径，mock 失败使其快速返回 */
    unsigned int m = 0;
    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&m)).will(returnValue(-1));

    char buf[4096];
    memset(buf, 0, sizeof(buf));
    size_t bufSize = sizeof(buf);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_NE(ret, 0);
    EXPECT_TRUE(UtLogContains("failed to fstat")) << "应输出 fstat 失败日志";
}

/**
 * @brief 验证 PassThrough 成功路径回填文件内容并释放 FILE*
 */
TEST_F(TopoAddrInfoTest, Ut_PassThrough_Success)
{
    const char* content = "0123456789abcdefghij"; /* 20 字节 */
    g_fileMock.passthrough_enable = 1;
    g_fileMock.tmpfile_content = content;

    char buf[4096];
    memset(buf, 0, sizeof(buf));
    size_t bufSize = sizeof(buf);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(bufSize, strlen(content));
    EXPECT_STREQ(buf, content);
    EXPECT_EQ(g_fileMock.mock_fclose_cnt, 1) << "成功路径必须 fclose，否则 FILE* 泄漏";
}

/**
 * @brief 验证 PassThrough 的文件大小判定边界：文件大小恰好等于 bufSize 时仍成功
 */
TEST_F(TopoAddrInfoTest, Ut_PassThrough_FileSizeEqualsBufSize)
{
    const char* content = "0123456789abcdefghij";
    g_fileMock.passthrough_enable = 1;
    g_fileMock.tmpfile_content = content;

    char buf[4096];
    memset(buf, 0, sizeof(buf));
    size_t bufSize = strlen(content);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(bufSize, strlen(content));
    EXPECT_EQ(g_fileMock.mock_fclose_cnt, 1);
}

/**
 * @brief 验证 PassThrough 的文件大小判定边界：文件大小超出 bufSize 时报错并释放 FILE*
 */
TEST_F(TopoAddrInfoTest, Ut_PassThrough_FileSizeExceedsBufSize)
{
    g_fileMock.passthrough_enable = 1;
    g_fileMock.tmpfile_content = "0123456789abcdefghij"; /* 20 字节 */

    /* PassThrough 失败后 TopoAddrInfoGet 回退到 mainboard_id 路径，mock 失败使其快速返回 */
    unsigned int m = 0;
    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&m)).will(returnValue(-1));

    char buf[4];
    memset(buf, 0, sizeof(buf));
    size_t bufSize = sizeof(buf);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);

    EXPECT_NE(ret, 0);
    EXPECT_TRUE(UtLogContains("exceeds buffer size")) << "应输出文件超限错误日志";
    EXPECT_EQ(g_fileMock.mock_fclose_cnt, 1);
}

/**
 * @brief 验证 mainboard_id 未命中时 TopoAddrInfoGet 返回 -1 并输出 WARN
 */
TEST_F(TopoAddrInfoTest, Ut_TopoAddrInfoGet_MainboardIdNotFound)
{
    /* 0xFF 不在 g_get_rootinfo_func_table 中 */
    unsigned int m = 0xFF;
    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&m)).will(returnValue(0));

    char buf[4096];
    memset(buf, 0, sizeof(buf));
    size_t bufSize = sizeof(buf);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_EQ(ret, -1);
    EXPECT_TRUE(UtLogContains("no get_rootinfo func")) << "未命中应输出 WARN 日志";
}

/**
 * @brief 验证 get_rootinfo 函数失败时 TopoAddrInfoGet 返回错误并输出 ERR
 */
TEST_F(TopoAddrInfoTest, Ut_TopoAddrInfoGet_GetRootinfoFail)
{
    unsigned int m = MAIN_BOARD_ID_CARD_2PMESH;
    char drv_path[256] = "/usr/local/Ascend2";
    dcmi_urma_eid_info_t eidList[MAX_EID_NUM];
    hex32_to_bin16("000000000000000000100000dfdf0020", eidList[0].eid.raw);
    hex32_to_bin16("000000000000000000100000dfdf0028", eidList[1].eid.raw);
    hex32_to_bin16("000000000000000000100000dfdf0030", eidList[2].eid.raw);
    hex32_to_bin16("000000000000000000100000dfdf0051", eidList[3].eid.raw);
    size_t eidNum = 4;

    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&m)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drv_path, strlen(drv_path)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(hal_get_eid_list_by_phy_id)
        .stubs()
        .with(mockcpp::any(), outBoundP(eidList, eidNum * sizeof(dcmi_urma_eid_info_t)), outBoundP(&eidNum))
        .will(returnValue(0));

    /* strcpy_s stub 恒返 0，注入失败返回值使 GetCardRankInfo 写入 rootinfo 失败 */
    MOCKER(strcpy_s).stubs().will(returnValue(1));

    char buf[4096];
    memset(buf, 0, sizeof(buf));
    size_t bufSize = sizeof(buf);
    int ret = TopoAddrInfoGet(0, buf, &bufSize);
    EXPECT_NE(ret, 0);
    EXPECT_TRUE(UtLogContains("get_rootinfo func failed")) << "应输出 get_rootinfo 函数失败 ERR 日志";
}

/**
 * @brief 验证 TopoLogInit 多线程并发调用不崩溃（pthread_mutex 保证线程安全）
 */
TEST_F(TopoAddrInfoTest, Ut_TopoLogInit_ConcurrentSafe)
{
    /* 置空函数指针以触发首次初始化：SetUp 的预设值会使所有线程
     * 在判空检查处早退，测不到锁内 dlopen/dlsym 的初始化竞争 */
    g_topo_DlogRecord = NULL;
    g_topo_CheckLogLevel = NULL;

    /* TopoLogInit 内部 dlopen 真实 libunified_dlog.so，结果不可控，
     * 只验证并发调用无数据竞争 */
    constexpr int threadNum = 16;
    std::vector<std::thread> ts;
    std::vector<int> results(threadNum, 0);
    for (int i = 0; i < threadNum; i++) {
        ts.emplace_back([&results, i] {
            TopoLogInit();
            results[i] = 1;
        });
    }
    for (int i = 0; i < threadNum; i++) {
        ts[i].join();
        EXPECT_EQ(results[i], 1) << "线程 " << i << " 应正常完成";
    }
    /* dlopen 成败不影响两个函数指针的赋值一致性 */
    bool dlogNull = (g_topo_DlogRecord == NULL);
    bool chkNull = (g_topo_CheckLogLevel == NULL);
    EXPECT_EQ(dlogNull, chkNull) << "两个函数指针应保持一致（同为 NULL 或同非 NULL）";
}

TEST_F(TopoAddrInfoTest, ut_init)
{
    int ret = load_dcmi();
    EXPECT_TRUE(ret == 0); // check already initialized
    EXPECT_TRUE(ret == 0); // test init
    MOCKER(hal_dlopen).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(mock_dlopen));
    MOCKER(hal_dlsym).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(mock_dlsym));
    reinit();
    ret = load_dcmi();
    EXPECT_TRUE(ret == 0);
}

/*
 * 验证GetUBEntityByFilter能够选中正确的UB Entity
 */
TEST_F(TopoAddrInfoTest, test_ub_entity_filter)
{
    UEList ueList;
    memset_s(&ueList, sizeof(UEList), 0x00, sizeof(UEList));
    hex32_to_bin16("000000000000020000100000df000101", ueList.ueList[0].eidList[0].eid.raw);
    ueList.ueList[0].eidNum = 1;
    // 定义一个iodie = 0  UE ID对应位置为a的UB_RTP EID,  由于是RTP，不应被选中
    hex32_to_bin16("0000000000080a8000100000df001101", ueList.ueList[1].eidList[0].eid.raw);
    ueList.ueList[1].eidNum = 1;
    // 定义一个iodie = 0  UE ID=4 的EID。 预期这个UB Entity被选中
    hex32_to_bin16("000000000001040000100000df001200", ueList.ueList[2].eidList[0].eid.raw);
    ueList.ueList[2].eidNum = 1;
    hex32_to_bin16("000000000000020000100000df001200", ueList.ueList[3].eidList[0].eid.raw);
    ueList.ueList[3].eidNum = 1;
    ueList.ueNum = 4;

    // 过滤iodie 0， 最大UE ID的UB entity， 类型为mesh, 预期过滤到ueList中下标为2的ub entity
    const UBEntity* ue = GetUBEntityByFilter(&ueList, 0, MAX_UE_ID, UE_TYPE_MESH);
    EXPECT_EQ(ue, &ueList.ueList[2]);
}
