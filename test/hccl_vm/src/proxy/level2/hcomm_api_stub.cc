/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "HCOMM_STUB"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include "operation_data/operation_data_ops.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_channel.h"
#include "hcomm_primitives.h"
#include "hccl_proxy_common.h"
#include "sim_log.h"
#include "store_sim_run_mode.h"
#include "ibv_dpu_stub.h"

namespace {
// HcclDfxOpInfoCompat 布局复刻 (hccl 仓 hccl_res_dl.h / CANN hccl_diag.h,
// CommAbiHeader 见 hcomm_res_defs.h): header(16B) + begin/endTime(16B) +
// 5×u32(20B) + pad(4B) + dataCount(8B) + root(4B) → algTag 偏移 68, 长 288
// (HCOMM_ALG_TAG_LENGTH)。结构体 algTag 之后的字段与本桩无关, 不复刻;
// 业务侧头文件漂移由 algTag 内容校验兜底 (不匹配则跳过截取, 不影响转发)。
struct CommAbiHeaderReplica {
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
};
struct HcclDfxOpInfoReplica {
    CommAbiHeaderReplica header;
    uint64_t beginTime;
    uint64_t endTime;
    uint32_t opMode;
    uint32_t opType;
    uint32_t reduceOp;
    uint32_t dataType;
    uint32_t outputType;
    uint64_t dataCount;
    uint32_t root;
    char algTag[288];
};
static_assert(
    offsetof(HcclDfxOpInfoReplica, algTag) == 68,
    "HcclDfxOpInfoCompat.algTag offset drifted from hccl_res_dl.h layout");

// algTag 格式 (hccl 仓 SetOpParamAlgTag/BuildCcuExtraTag):
//   "{opType_commName}_{algName}_{host|device}[_{dataType}[_{reduceOp}[_r{root}]]]"
//   图模式 CCU: "Graph_{algName}_{launchMode}..." (无 commName 段)
// 算法名本身不含下划线; 借 commId 参数定位前缀、launchMode 后缀校验结构,
// 过滤 op-begin 等不含算法名的注册 (如 "{opType}_{comm}_{dtype}_{count}")
bool ExtractAlgNameFromDfxTag(const char* tag, const char* commId, std::string& algName)
{
    if (tag == nullptr) {
        return false;
    }
    const std::string s(tag);
    if (s.empty() || s.size() >= 288) {
        return false;
    }
    size_t start = std::string::npos;
    if (s.rfind("Graph_", 0) == 0) {
        start = 6; // strlen("Graph_")
    } else {
        if (commId == nullptr || commId[0] == '\0') {
            return false;
        }
        const std::string pat = std::string("_") + commId + "_";
        const size_t p = s.find(pat);
        if (p == std::string::npos) {
            return false;
        }
        start = p + pat.size();
    }
    const size_t end = s.find('_', start);
    if (end == std::string::npos) {
        return false;
    }
    const std::string rest = s.substr(end + 1);
    // 后随 launchMode ("host"/"device"), 借此确认中间字段确为算法名
    if (rest.rfind("host", 0) != 0 && rest.rfind("device", 0) != 0) {
        return false;
    }
    algName = s.substr(start, end - start);
    return !algName.empty() && algName.size() <= 128;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 按标签获取设备工作区内存（level2 透传桩）。
 *
 * 默认（level2）模式：RTLD_NEXT 透传真实 libhcomm 实现，成功后把 host 地址
 * 转换为 device 映射地址返回；level1 模式由 level1 DPU 控制面桩接管，
 * 本桩防御性拒绝（防误透传真实 libhcomm 导致模型冲突）。
 *
 * @param comm       输入：通信域句柄。
 * @param memTag     输入：内存标签。
 * @param size       输入输出：请求/实际尺寸。
 * @param addr       输出：内存地址（device 映射地址）。
 * @param newCreated 输出：是否新创建。
 * @return HcclResult 成功返回 HCCL_SUCCESS，否则返回对应错误码。
 */
HcclResult HcclDevMemAcquire(HcclComm comm, const char* memTag, uint64_t* size, void** addr, bool* newCreated)
{
    // level1 模式下本接口由 libhccl_proxy_level1.so 的 DPU 控制面桩接管
    // （dpu_control_stub.cc：真实分配共享内存 + 登记 + 拉起 DPU 执行器），
    // 本透传桩防御性拒绝，避免误透传真实 libhcomm 导致模型冲突。
    const char* vmLevel = std::getenv("HCCL_VM_LEVEL");
    if (vmLevel != nullptr && std::strcmp(vmLevel, "1") == 0) {
        HCCL_VM_ERROR("HcclDevMemAcquire is served by level1 stub in level1 mode; level2 passthrough rejected");
        return HCCL_E_NOT_SUPPORT;
    }

    // 若符号解析错误(如 RTLD_NEXT 把符号解析回本桩), 重入会在此被立即打断,
    // 避免无限递归把栈打爆(stack overflow)。
    static thread_local bool inCall = false;
    if (inCall) {
        HCCL_VM_ERROR("HcclDevMemAcquire re-entered (self-recursion detected), abort.");
        return HCCL_E_INTERNAL;
    }
    inCall = true;

    HCCL_VM_INFO(
        "[HcclDevMemAcquire] comm={:p}, memTag={}, size_ptr={:p}, addr_ptr={:p}, newCreated_ptr={:p}",
        static_cast<void*>(comm), (memTag != nullptr) ? memTag : "nullptr", static_cast<void*>(size),
        static_cast<void*>(addr), static_cast<void*>(newCreated));

    using RealFunc = HcclResult (*)(HcclComm comm, const char* memTag, uint64_t* size, void** addr, bool* newCreated);

    // RTLD_NEXT 无法穿透 torch_npu 等场景中以 RTLD_LOCAL 方式加载的 libhcomm.so,
    // 失败时按 SONAME 显式加载后重试 (libhcomm.so 导出 HcclDevMemAcquire).
    const auto resolvedFn = reinterpret_cast<RealFunc>(sim::DlsymRealWithFallback(__func__, "libhcomm.so"));
    if (resolvedFn == nullptr) {
        inCall = false;
        return HCCL_E_INTERNAL;
    }

    // 校验解析结果不是 CheckerL2 自己打桩的符号(位于 libhccl_proxy_* 内),
    // 而是来自某个真实库。若被符号插入回 proxy 自己或兄弟 .so, 说明解析错误, 直接拒绝。
    Dl_info resolvedInfo{};
    if (dladdr(reinterpret_cast<void*>(resolvedFn), &resolvedInfo) == 0 || resolvedInfo.dli_fname == nullptr
        || std::string(resolvedInfo.dli_fname).find("libhccl_proxy") != std::string::npos) {
        HCCL_VM_ERROR(
            "HcclDevMemAcquire resolvedFn comes from a proxy stub ({}) , abort to avoid self-call.",
            (resolvedInfo.dli_fname != nullptr) ? resolvedInfo.dli_fname : "unknown");
        inCall = false;
        return HCCL_E_INTERNAL;
    }

    // 转发调用真实底层函数
    HcclResult ret = resolvedFn(comm, memTag, size, addr, newCreated);
    if (ret != HCCL_SUCCESS) {
        HCCL_VM_ERROR("HcclDevMemAcquire realFn failed, ret={}", static_cast<int>(ret));
        inCall = false;
        return ret;
    }

    // 仅在真实函数成功时，将*addr(host进程可访问地址)转化为device进程可访问地址
    *addr = sim::GetDevMapperAddrByHostPtr(*addr);
    if (*addr == nullptr) {
        inCall = false;
        return HCCL_E_PTR;
    }

    inCall = false;
    return ret;
}

/**
 * @brief 按通信域 ID 注册算子级 DFX 信息 (劫持: algTag 截取 + 转发)。
 *
 * 业务侧 (hccl 仓 op_common.cc) 在算法选择完成后逐算子调用本接口, 入参
 * HcclDfxOpInfoCompat.algTag 携带所选算法名。Level=2 模式 (默认) 仅挂
 * level0+level2, level1 的 dfx_stub 不在进程内; 本桩以 level2 (LD_PRELOAD)
 * 强符号抢占解析 (先于业务经 python 插件链加载的 libhcomm.so), 截取
 * algTag 中的算法名回填 opDetails.algorithmName 后转发真实实现:
 *   1. 解析 algTag → sim::UpdateOpDetailAlgName (last-wins, 与
 *      log_stub.cc 日志截取互为备份, 二者写入同值)
 *   2. DlsymRealWithFallback (RTLD_NEXT + libhcomm.so SONAME 回退) 转发,
 *      返回值透传, 保持原有调用流程与语义
 * Level=1 模式下 level1 的 dfx_stub 先于本桩参与解析, 本桩被遮蔽 (行为
 * 同旧版), 截取退回日志路径兜底。
 */
HcclResult HcclDfxRegOpInfoByCommId(char* commId, void* hcclDfxOpInfo)
{
    // 防自调回环: 转发目标解析异常时重入会在此被打断, 避免无限递归
    static thread_local bool inCall = false;
    if (inCall) {
        HCCL_VM_ERROR("{} re-entered (self-recursion detected), abort.", __func__);
        return HCCL_E_INTERNAL;
    }
    inCall = true;

    if (commId == nullptr) {
        inCall = false;
        HCCL_VM_ERROR("{}: commId is nullptr", __func__);
        return HCCL_E_PTR;
    }

    // 仅截取呈现信息, 不改变任何调用语义; 结构/内容校验不过则静默跳过
    if (hcclDfxOpInfo != nullptr) {
        const auto* info = static_cast<const HcclDfxOpInfoReplica*>(hcclDfxOpInfo);
        std::string algName;
        if (ExtractAlgNameFromDfxTag(info->algTag, commId, algName)) {
            (void)sim::operation::UpdateOpDetailAlgName(algName);
            HCCL_VM_INFO("{}: commId={}, algTag algName={} captured", __func__, commId, algName);
        }
    }

    // 转发真实 libhcomm 实现
    using RealFunc = HcclResult (*)(char*, void*);
    const auto realFn = reinterpret_cast<RealFunc>(sim::DlsymRealWithFallback(__func__, "libhcomm.so"));
    if (realFn == nullptr) {
        inCall = false;
        HCCL_VM_ERROR("{}: real libhcomm impl not found, skip forward (dfx info dropped)", __func__);
        return HCCL_SUCCESS;
    }
    // 解析结果须来自真实库而非 proxy 桩 (防符号插入回环)
    Dl_info resolvedInfo{};
    if (dladdr(reinterpret_cast<void*>(realFn), &resolvedInfo) == 0 || resolvedInfo.dli_fname == nullptr
        || std::string(resolvedInfo.dli_fname).find("libhccl_proxy") != std::string::npos) {
        inCall = false;
        HCCL_VM_ERROR(
            "{} realFn comes from a proxy stub ({}), abort to avoid self-call.", __func__,
            (resolvedInfo.dli_fname != nullptr) ? resolvedInfo.dli_fname : "unknown");
        return HCCL_E_INTERNAL;
    }

    inCall = false;
    return realFn(commId, hcclDfxOpInfo);
}

// 评估模式+HostDPU侧专用劫持: 记录(channel, idx)供ibv层建立槽位映射，正常模式纯转发
int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    using RealFunc = int32_t (*)(ThreadHandle, ChannelHandle, uint32_t);
    static RealFunc realFn = reinterpret_cast<RealFunc>(dlsym(RTLD_NEXT, "HcommChannelNotifyRecordOnThread"));
    if (realFn == nullptr) {
        HCCL_VM_ERROR("HcommChannelNotifyRecordOnThread dlsym failed");
        return 1; // HCCL_E_PARA
    }
    if (!sim::IsEstimateMode()) {
        return realFn(thread, channel, remoteNotifyIdx);
    }
    HCCL_VM_INFO("prim: record ch={:x} idx={}", channel, remoteNotifyIdx);
    DpuSetNotifyRecordContext(channel, remoteNotifyIdx);
    int32_t ret = realFn(thread, channel, remoteNotifyIdx);
    DpuClearNotifyContext();
    return ret;
}

int32_t HcommWriteWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    using RealFunc = int32_t (*)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, uint32_t);
    static RealFunc realFn = reinterpret_cast<RealFunc>(dlsym(RTLD_NEXT, "HcommWriteWithNotifyOnThread"));
    if (realFn == nullptr) {
        HCCL_VM_ERROR("HcommWriteWithNotifyOnThread dlsym failed");
        return 1;
    }
    if (!sim::IsEstimateMode()) {
        return realFn(thread, channel, dst, src, len, remoteNotifyIdx);
    }
    DpuSetNotifyRecordContext(channel, remoteNotifyIdx);
    int32_t ret = realFn(thread, channel, dst, src, len, remoteNotifyIdx);
    DpuClearNotifyContext();
    return ret;
}

int32_t HcommWriteWithNotifyNbiOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    using RealFunc = int32_t (*)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, uint32_t);
    static RealFunc realFn = reinterpret_cast<RealFunc>(dlsym(RTLD_NEXT, "HcommWriteWithNotifyNbiOnThread"));
    if (realFn == nullptr) {
        HCCL_VM_ERROR("HcommWriteWithNotifyNbiOnThread dlsym failed");
        return 1;
    }
    if (!sim::IsEstimateMode()) {
        return realFn(thread, channel, dst, src, len, remoteNotifyIdx);
    }
    DpuSetNotifyRecordContext(channel, remoteNotifyIdx);
    int32_t ret = realFn(thread, channel, dst, src, len, remoteNotifyIdx);
    DpuClearNotifyContext();
    return ret;
}

int32_t
HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    using RealFunc = int32_t (*)(ThreadHandle, ChannelHandle, uint32_t, uint32_t);
    static RealFunc realFn = reinterpret_cast<RealFunc>(dlsym(RTLD_NEXT, "HcommChannelNotifyWaitOnThread"));
    if (realFn == nullptr) {
        HCCL_VM_ERROR("HcommChannelNotifyWaitOnThread dlsym failed");
        return 1;
    }
    if (!sim::IsEstimateMode()) {
        return realFn(thread, channel, localNotifyIdx, timeOut);
    }
    HCCL_VM_INFO("prim: wait ch={:x} idx={}", channel, localNotifyIdx);
    DpuSetNotifyWaitContext(channel, localNotifyIdx);
    int32_t ret = realFn(thread, channel, localNotifyIdx, timeOut);
    DpuClearNotifyContext();
    return ret;
}

int32_t
HcommChannelNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx)
{
    using RealFunc = int32_t (*)(ThreadHandle, ChannelHandle, uint32_t);
    static RealFunc realFn
        = reinterpret_cast<RealFunc>(dlsym(RTLD_NEXT, "HcommChannelNotifyWaitOnThreadWithDefaultTimeout"));
    if (realFn == nullptr) {
        HCCL_VM_ERROR("HcommChannelNotifyWaitOnThreadWithDefaultTimeout dlsym failed");
        return 1;
    }
    if (!sim::IsEstimateMode()) {
        return realFn(thread, channel, localNotifyIdx);
    }
    HCCL_VM_INFO("prim: wait ch={:x} idx={}", channel, localNotifyIdx);
    DpuSetNotifyWaitContext(channel, localNotifyIdx);
    int32_t ret = realFn(thread, channel, localNotifyIdx);
    DpuClearNotifyContext();
    return ret;
}

// 评估模式: 通道创建拦截(RoCE通道), 创建后按HcclChannelDesc.notifyNum预填槽位映射; 正常模式纯转发
HcclResult HcclChannelAcquire(
    HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    using RealFunc = HcclResult (*)(HcclComm, CommEngine, const HcclChannelDesc*, uint32_t, ChannelHandle*);
    static RealFunc realFn = reinterpret_cast<RealFunc>(dlsym(RTLD_NEXT, "HcclChannelAcquire"));
    if (realFn == nullptr) {
        HCCL_VM_ERROR("HcclChannelAcquire dlsym failed");
        return HCCL_E_INTERNAL;
    }
    HcclResult ret = realFn(comm, engine, channelDescs, channelNum, channels);
    if (sim::IsEstimateMode() && ret == HCCL_SUCCESS && channels != nullptr && engine == COMM_ENGINE_CPU) {
        for (uint32_t i = 0; i < channelNum; i++) {
            uint32_t notifyNum = (channelDescs != nullptr) ? channelDescs[i].notifyNum : 0;
            DpuOnChannelCreatedEstimate(channels[i], notifyNum);
        }
    }
    return ret;
}

#ifdef __cplusplus
}
#endif // __cplusplus
