/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_KERNEL_H
#define HCOMM_CCU_KERNEL_H

#include <cstdint>
#include <functional>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "ccu_task_param_v1.h"

#include "ccu_kernel_resource.h"
#include "ccu_instr_info_v1.h"
#include "ccu_rep_context_v1.h"

#include "ccu_funccall_v1.h"

#include "hccl_types.h"
#include "hcomm_primitives.h"
#include "ccu_datatype_v1.h"
#include "ccu_interface_assist_v1.h"

#include "ccu_res_repo.h"
#include "ccu_dev_mgr_imp.h" // CcuVersion

#include "ccu_rep_jump_v1.h"
#include "ccu_rep_jumplabel_v1.h"
#include "ccu_rep_loopblock_v1.h"
#include "ccu_rep_funcblock_v1.h"
#include "ccu_rep_addr_funcblock_v1.h"
#include "ccu_rep_addr_funccall_v1.h"
#include "ccu_assist_pub.h"

#ifndef CCU_PROFILING // 和hccl仓兼容性使用
#define CCU_PROFILING
#endif
#include "ccu_types.h"
#include "ccu_profiling.h"
#include "ccu_rep_handle_repo.h"

namespace hcomm {

// Kernel 查询信息结构体，由 CcuKernel::GetCcuKernelInfo 在锁内填充。
// 后续新增查询字段时只需在此结构体中添加，无需新增接口。
struct CcuKernelInfo {
    uint32_t maxTaskArgsNum{0}; // taskArgs数组所需的元素个数，注册期间由LoadArg的argId推算: max(argId)+1，空则为0
};

// 跨kernel调用命名func时caller与callee共用的寄存器。两侧分属不同kernel与mission，各自私有的
// 传参池和per-mission返回地址寄存器物理Id并不相同，必须统一到这批按die分配的共享Xn上。
struct CcuFuncAddr {
    std::vector<CcuRep::Variable> argVars; // 传参寄存器，下标即参数序号
    CcuRep::Variable retVar;               // 返回地址寄存器
    CcuRep::Variable jumpScratch;          // 调用点承载目标 pc 的跳转暂存寄存器
};

class CcuKernel : public CcuRep::CcuRepContext {
public:
    CcuKernel() = default;
    ~CcuKernel() override = default;

    HcclResult SetupProfilingInfo(const char* kernelFuncName) { return profiling_.SetupProfilingInfo(kernelFuncName); }
    HcclResult ApplyDieFromChannels();
    HcclResult ValidateAndApplyDie(uint32_t targetDieId);

    CcuResReq GetResourceRequest();
    CcuResRepository& GetResRepository();
    CcuRepResource& GetResource();
    CcuSharedResource& GetExportedRes();
    CcuSharedResource& GetImportedRes();

    void SetResRepository(const CcuResRepository& resRepo);
    void SetCascCntBlock(const std::unordered_map<HcommCcuCascCntHandle, CntXnBlock>& cascCntBlocks);
    // 记录 rep 序列引用到的级联计数器句柄, 供注册期按句柄归属实例解析出具体 block
    void AddCascCntHandle(HcommCcuCascCntHandle cntHandle);
    std::unordered_set<HcommCcuCascCntHandle> GetCascCntHandles();
    void SetInstrId(uint32_t instrId);
    HcclResult GetCascCntBlock(HcommCcuCascCntHandle cntHandle, CntXnBlock& cascCntBlock);
    uint32_t GetInstrId() const;
    uint32_t GetInstrCount();
    // 统计会翻译出 CKE 写者微码 (profiling -> setcke / 非 profiling -> clearcke) 的三种 wait 类 rep
    // 个数 (含 block 子 rep), 用于按 CCU_CKE_RAW_LATENCY 为每个此类 rep 预留指令空间.
    // 每个此类 rep 只发射 1 条 setcke / clearcke.
    uint32_t GetRepNeedToAddLatency() const;
    // 统计会翻译出 LoadX/StoreX/ClearX (half-rtt 写后读写者) 的 rep 个数
    // = #(LOAD_ADD_IMM + STORE_ADD_IMM + CASC_CNT_CLEAR) (含 block 子 rep), 用于按 CCU_XN_RAW_LATENCY
    // 为每个此类 rep 预留指令空间. 每个此类 rep 恰翻译出 1 条 LoadX/StoreX/ClearX, 后端优化最多为其
    // 后续读者补 (CCU_XN_RAW_LATENCY - 1) 条 NOP. 仅 CCU_V2 生效.
    uint32_t GetLsxRepReserveCount() const;
    void SetCcuInstrInfo(const CcuRep::CcuInstrInfo& instrInfo);
    // 获取下发硬件的最终指令信息 (V2 场景为后端优化后的序列). DFX 异常回查需据此把硬件 currentIns
    // (优化后坐标) 通过 optToRawLocalIndex 翻译回 Rep 区间 (优化前坐标).
    const CcuRep::CcuInstrInfo& GetCcuInstrInfo() const { return instrInfo_; }

    CcuResult GeneTaskParams(const uint64_t* taskArgs, uint32_t argsNum, std::vector<CcuTaskParam>& taskParams);

    void SetInsGenerater(CcuRep::CcuInsGeneratorBase* insGeneratorBase);
    void SetCcuVersion(CcuVersion version) { ccuVersion_ = version; }
    // 该友元函数用于在context类外创建Variable并被context内的资源管理器管理
    friend CcuRep::Variable CcuRep::CreateVariable(CcuRep::CcuRepContext* context);

    HcclResult AddProfilingInfo(
        const ChannelHandle* channels, uint32_t channelNum, HcclDataType dataType, HcclDataType outputDataType,
        HcclReduceOp opType, const std::string& opName)
    {
        return profiling_.AddProfilingInfo(channels, channelNum, dataType, outputDataType, opType, opName);
    }

    // 双路 profiling 收集：ccuStaticInfo 为 SQE/WaitCKE 静态部分（shared_ptr 0 拷贝共享），
    // ccuDetailInfo 为 LoopGroup 动态部分（深拷贝）；空时为 nullptr。
    HcclResult GetCcuProfilingInfo(
        const uint64_t* taskArgs, uint32_t argSize, std::shared_ptr<std::vector<CcuProfilingInfo>>& ccuStaticInfo,
        std::shared_ptr<std::vector<CcuProfilingInfo>>& ccuDetailInfo)
    {
        return profiling_.GetCcuProfilingInfo(taskArgs, argSize, ccuStaticInfo, ccuDetailInfo, GetInstrId());
    }

    // 按 isSaveProfilingData 开关决定是否收集（为 false 时直接返回不收集）
    HcclResult ConstructProfilingInfo(
        const uint64_t* taskArgs, uint32_t argNum, CcuProfiling::CcuDfxResult& result, bool isSaveProfilingData)
    {
        return profiling_.ConstructProfilingInfo(taskArgs, argNum, result, isSaveProfilingData, GetInstrId());
    }

    const std::vector<CcuProfilingInfo>& GetAllCcuProfilingInfo() { return profiling_.GetAllCcuProfilingInfo(); }

    // process const values
    std::unordered_map<uint64_t, CcuRep::Variable>& GetConstValue2VarMap() { return constValue2VarMap; }
    HcclResult
    Add2ConstValue2VarMap(const std::vector<uint64_t>& values, const std::vector<uint64_t>& secretValues = {});
    bool IsSecConstValue(uint64_t value) const;

    const std::unordered_set<ChannelHandle>& GetChannels() { return channels_; }

    const std::unordered_set<uint32_t>& GetDeclaredLocXns() const { return declaredLocXns_; }

    struct PinnedGroupEntry {
        CcuRep::Variable baseVar;
        uint16_t count;
    };
    const std::vector<PinnedGroupEntry>& GetPinnedRegGroups() const { return pinnedRegGroups_; }
    void AddPinnedRegGroup(const CcuRep::Variable& baseVar, uint16_t count);

    // 句柄表：构造 rep 时经此解析句柄，登记仍由本类的 Alloc 接口完成
    CcuRepHandleRepo& Repo() { return repo_; }

    // 供 ccu_arithmetic、ccu_trans、ccu_sync、ccu_load 的自由函数构造 rep 时调用。
    // module-internal，非 Kernel 公共 API。
    CcuRep::Variable CreateVariableForRepBuild();
    CcuRep::LocalNotify CreateLocalNotifyForRepBuild();
    CcuResult LatchBodyErrorForRepBuild(CcuResult err);
    void AddChannelForRepBuild(ChannelHandle channel);
    void AddLoadArgUsedForRepBuild(uint32_t argId);

public:
    // Alloc 相关接口
    CcuResult VariableAlloc(CcuVariableHandle* varHandle);
    CcuResult AddressAlloc(CcuAddressHandle* addrHandle);
    CcuResult EventAlloc(CcuEventHandle* eventHandle);
    CcuResult BufferAlloc(CcuBufferHandle* bufHandle);
    CcuResult
    LocalAddrAlloc(CcuLocalAddrHandle* localAddrHandle, CcuAddressHandle* addrHandle, CcuVariableHandle* tokenHandle);
    CcuResult RemoteAddrAlloc(
        CcuRemoteAddrHandle* remoteAddrHandle, CcuAddressHandle* addrHandle, CcuVariableHandle* tokenHandle);
    CcuResult BlockVariableAlloc(CcuVariableHandle* varHandles, uint32_t count);
    CcuResult BlockEventAlloc(CcuEventHandle* eventHandles, uint32_t count);
    CcuResult BlockBufferAlloc(CcuBufferHandle* bufHandles, uint32_t count);
    CcuResult VariableCreateByChannel(ChannelHandle channel, uint32_t varIndex, CcuVariableHandle* varHandle);
    CcuResult VariableCreateByAcquire(CcuVariableHandle acqHandle, uint32_t index, CcuVariableHandle* varHandle);
    CcuResult EventCreateByAcquire(CcuEventHandle acqHandle, uint32_t index, CcuEventHandle* eventHandle);

    // 在锁内填充 Kernel 查询信息；当前包含 maxTaskArgsNum。
    CcuResult GetCcuKernelInfo(CcuKernelInfo& info) const;

    CcuResult IfBegin(CcuVariableHandle varHandle, uint64_t immediate, CcuConditionType condType, const char* label);
    CcuResult
    IfBeginVar(CcuVariableHandle lhsHandle, CcuVariableHandle rhsHandle, CcuConditionType condType, const char* label);
    CcuResult IfElse(const char* label);
    CcuResult IfEnd(const char* label);

    void IfLabelStackPush(const char* label);
    void IfLabelStackMarkBodyDone();
    const char* IfLabelStackPopForElse();
    bool IfLabelStackTopIsClosable();
    const char* IfLabelStackPop();

    void DoWhileLabelStackPush(const char* label);
    const char* DoWhileLabelStackPopForWhile();

    void FlushClosablePendingIfs();
    void Append(std::shared_ptr<CcuRep::CcuRepBase> rep) override;
    CcuResult WhileBegin(CcuVariableHandle varHandle, uint64_t immediate, CcuConditionType condType, const char* label);
    CcuResult WhileBeginVar(
        CcuVariableHandle lhsHandle, CcuVariableHandle rhsHandle, CcuConditionType condType, const char* label);
    CcuResult WhileEnd(const char* label);
    CcuResult DoWhileBegin(const char* label);
    CcuResult DoWhileEnd(CcuVariableHandle varHandle, uint64_t immediate, CcuConditionType condType, const char* label);
    CcuResult DoWhileEndVar(
        CcuVariableHandle lhsHandle, CcuVariableHandle rhsHandle, CcuConditionType condType, const char* label);

    CcuResult LoopCreate(CcuLoop* loop);
    CcuResult LoopBodyEnter(CcuLoop loop);
    CcuResult LoopBodyExit(CcuLoop loop);
    // 建组/加环统一以版本化 cfg 为准;旧 config 的转换在 C ABI 适配层完成。
    CcuResult LoopGroupCreate(CcuLoopGroup* group, uint32_t maxLoopNum, const CcuLoopGroupCfg* cfg);
    CcuResult LoopGroupCreateFromVar(
        CcuLoopGroup* group, uint32_t maxLoopNum, CcuVariableHandle parallelVarHandle,
        CcuVariableHandle offsetVarHandle);
    CcuResult LoopGroupCreateFromVarV2(
        CcuLoopGroup* group, uint32_t maxLoopNum, CcuVariableHandle parallelVarV2Handle,
        CcuVariableHandle offsetVarV2Handle, CcuVariableHandle varOffsetVarHandle);
    CcuResult LoopGroupAddLoop(CcuLoopGroup group, CcuLoop loop, const CcuLoopCfg* cfg);
    CcuResult LoopGroupAddLoopFromVar(CcuLoopGroup group, CcuLoop loop, CcuVariableHandle loopParamVarHandle);
    CcuResult LoopGroupAddLoopFromVarV2(
        CcuLoopGroup group, CcuLoop loop, CcuVariableHandle iterNumVarHandle, CcuVariableHandle addrOffsetVarHandle,
        CcuVariableHandle ctxIdVarHandle);

    CcuResult FuncBlockLookup(const void* funcPtr, uint64_t* outHandle);
    CcuResult FuncBlockBegin(const void* funcPtr, uint64_t* outHandle);
    CcuResult FuncBlockEnd(uint64_t handle);
    CcuResult FuncDefineInArg(uint64_t handle, CcuVariableHandle formal);
    CcuResult FuncCall(uint64_t handle, const CcuVariableHandle* inArgs, uint32_t numIn);
    // 命名 func 在声明处即绑定到当前 kernel，name 就是它在本 kernel 内的唯一身份
    CcuResult FuncBlockBeginNamed(const char* name, uint64_t* outHandle);
    CcuResult FuncCallByAddr(CcuVariableHandle funcPc, const CcuVariableHandle* inArgs, uint32_t numIn);
    // 按 func 名查询入口指令 id，须在翻译之后调用
    CcuResult GetFuncPc(const char* name, uint64_t* pc) const;
    // 把本 die 的 func Addr 共享传参回填到组装期登记的命名 func 与间接调用点。
    // 组装期 die 尚未确定（资源先记在 die 0），因此必须等 die 落定后再绑定。
    CcuResult BindFuncAddr(const CcuFuncAddr* addr);
    // 常量寄存器确定后，把可寻址 func 自包含序言的指令数告知其 FuncBlock，供指令空间统计
    void UpdateAddressableFuncPreludeCount();

private:
    CcuResult EnsureLoopEnginePool(uint32_t maxLoopNum);
    // 登记一个要走 Addr 共享传参的调用点或命名 func 入口，入参个数超上限直接失败。
    CcuResult RegisterAddrUser(uint32_t numIn);
    // 新建 FuncBlock 并把录制点切进它，返回 handle；funcPtr 为空表示不按对象地址索引
    uint64_t OpenFuncBlock(const void* funcPtr);

    CcuResult ValidateTaskArgs(const uint64_t* taskArgs, uint32_t argsNum) const;
    CcuResult FillTaskParam(
        CcuTaskParam& param, uint32_t index, uint32_t seqNum, const uint64_t* taskArgs, uint32_t argsNum) const;

    CcuRep::Variable CreateJumpTargetVar();

    struct IfLabelEntry {
        const char* label{nullptr};
        bool bodyDone{false};
    };
    struct DoWhileLabelEntry {
        const char* label{nullptr};
        std::shared_ptr<CcuRep::CcuRepBlock> snapshotBlock{nullptr};
        size_t snapshotRepCount{0};
    };
    std::vector<IfLabelEntry> iflabelStack_;
    std::vector<DoWhileLabelEntry> doWhileLabelStack_;
    bool isFlushing_ = false;

    struct PendingIfContext {
        std::shared_ptr<CcuRep::CcuRepJumpLabel> elseLabel;
        std::shared_ptr<CcuRep::CcuRepJumpLabel> endLabel;
        bool hasElse{false};
    };

    struct PendingWhileContext {
        std::shared_ptr<CcuRep::CcuRepJumpLabel> beginLabel;
        std::shared_ptr<CcuRep::CcuRepJumpLabel> endLabel;
        CcuVariableHandle varHandle;
        uint64_t immediate;
        CcuConditionType condType;
    };

    struct PendingDoWhileContext {
        std::shared_ptr<CcuRep::CcuRepJumpLabel> beginLabel;
    };

    std::unordered_map<std::string, PendingIfContext> pendingIfCtx_{};
    std::unordered_map<std::string, PendingWhileContext> pendingWhileCtx_{};
    std::unordered_map<std::string, PendingDoWhileContext> pendingDoWhileCtx_{};

    std::unordered_set<uint32_t> loadArgUsedSet_{};

protected:
    // 使用channel中的Variable
    HcclResult CreateVariable(const ChannelHandle channel, uint32_t varIndex, CcuRep::Variable* var);
    CcuRep::Variable CreateExpectVar();
    CcuRep::LocalAddr CreateLocalAddr();
    CcuRep::RemoteAddr CreateRemoteAddr();

    HcclResult CreateBlockExecutor(const uint32_t count, CcuRep::Executor* ccuExes);

    // 控制逻辑
    // 宏定义IF、WHILE
    CcuRep::FuncCall Func(const std::string& label);
    CcuRep::FuncCall Func(const CcuRep::Variable& funcAddr);

private:
    CcuRep::Address CreateAddress();
    CcuRep::LocalAddr CreateLocalAddr(const CcuRep::Variable& token);

private:
    template <typename T>
    T CreateResAssist(std::array<std::vector<T>, CCU_MAX_IODIE_NUM>& resRecord);
    template <typename T>
    std::vector<T> CreateBlockResAssist(const uint32_t count, std::array<std::vector<T>, CCU_MAX_IODIE_NUM>& resRecord);

private:
    CcuRepResource res_{};
    CcuResRepository resRepo_{};
    CcuRepHandleRepo repo_{};

    std::mutex cascCntBlockMutex_;
    std::unordered_map<HcommCcuCascCntHandle, CntXnBlock> cascCntBlocks_{};
    std::unordered_set<HcommCcuCascCntHandle> cascCntHandles_{};

    std::unordered_set<ChannelHandle> channels_{};

    std::unordered_set<uint32_t> declaredLocXns_{};

    std::vector<PinnedGroupEntry> pinnedRegGroups_{};

    CcuRep::CcuInstrInfo instrInfo_{};

    CcuVersion ccuVersion_{CcuVersion::CCU_INVALID};

    // A5(CCU_V1) 专用：CCU_IF,CCU_ELSE 跳转共用一个 Xn；当前仅CCU_IF和CCU_ELSE复用，While和DoWhile暂未复用
    // A5 上 CCU_IF和ELSE的targetVar 仅用于LoadImdToXn + 紧接 Jump，生命周期不跨越其他指令，可安全复用。
    // A6不复用，保持每次独立分配的语义。
    std::unique_ptr<CcuRep::Variable> sharedJumpTargetVar_;

    CcuSharedResource exportedRes_{};
    CcuSharedResource importedRes_{};
    // 记录每个kernel所需常量，适用于A6场景
    std::unordered_map<uint64_t, CcuRep::Variable> constValue2VarMap;
    // 记录常量表中属于敏感信息(token)的值，装载到寄存器时用于打 secFlag 标记
    std::unordered_set<uint64_t> secConstValueSet_;

    struct LoopDescriptor {
        std::string label;
        std::shared_ptr<CcuRep::CcuRepLoopBlock> repLoopBlock;
        std::shared_ptr<CcuRep::CcuRepBlock> prevActiveBlock;
        bool bodyDefined{false};
    };

    struct VersionV2LoopRecord {
        CcuRep::Variable iterNumVar;
        CcuRep::Variable addrOffsetVar;
        CcuRep::Variable ctxIdVar;
    };

    struct LoopGroupDescriptor {
        CcuLoopGroupCfg config;
        uint64_t totalLoopNum{0};
        uint32_t loopCount{0};
        CcuRep::Variable parallelVar;
        CcuRep::Variable offsetVar;
        CcuRep::Variable xnOffsetVar;
        std::shared_ptr<CcuRep::CcuRepBase> bundleRep;
        bool isVarBased{false};
        bool isVersionV2{false};
        std::vector<VersionV2LoopRecord> versionV2Loops;
    };

    struct FuncDescriptor {
        const void* funcPtr{nullptr};
        std::string label;
        // 原 callfunc<>（匿名 func）用 repFuncBlock；命名 / 可寻址 func 用 repAddrFuncBlock，两条链路互不交叉
        std::shared_ptr<CcuRep::CcuRepFuncBlock> repFuncBlock;
        std::shared_ptr<CcuRep::CcuRepAddressableFuncBlock> repAddrFuncBlock;
        std::shared_ptr<CcuRep::CcuRepBlock> prevActiveBlock;
        bool bodyDefined{false};
        std::string name; // 非空即可被间接调用：入参改走中转寄存器，且可按名字查 pc
    };

    std::unordered_map<CcuLoop, LoopDescriptor> loopMap_;
    std::unordered_map<CcuLoopGroup, LoopGroupDescriptor> loopGroupMap_;
    CcuResult LookupLoopGroupAndLoop(
        CcuLoopGroup group, CcuLoop loop, const char* fnName, const char* createFnName, LoopGroupDescriptor*& grpDesc,
        LoopDescriptor*& loopDesc, uint32_t& loopIdx);
    uint32_t loopHandleCounter_{0};
    uint32_t loopGroupHandleCounter_{0};
    uint32_t loopBodyDepth_{0};

    std::unordered_map<uint64_t, FuncDescriptor> funcMap_;
    std::unordered_map<const void*, uint64_t> funcInstanceMap_;
    std::unordered_map<std::string, uint64_t> funcNameMap_;
    // 组装期登记、die 落定后统一回填 Addr 共享传参的命名 func 入口与间接调用点
    std::vector<std::shared_ptr<CcuRep::CcuRepAddressableFuncBlock>> addrFuncBlocks_;
    std::vector<std::shared_ptr<CcuRep::CcuRepAddrFuncCall>> addrFuncCalls_;
    uint64_t funcHandleCounter_{0};
    bool inFuncBody_{false};
    // loop/func body 内首个非法错误的粘性闩，退出 body 时上抛（void body 无法直接回传）。
    CcuResult bodyError_{CcuResult::CCU_SUCCESS};

    std::unordered_map<CcuLoopExecutors, std::vector<CcuRep::Executor>> loopEnginePools_;
    uint32_t loopEnginePoolCounter_{0};

    CcuProfiling profiling_{*this};
};

CcuResult GetChannelIdByHandle(ChannelHandle channel, uint32_t& channelId);

} // namespace hcomm

#endif // HCOMM_CCU_KERNEL_H
