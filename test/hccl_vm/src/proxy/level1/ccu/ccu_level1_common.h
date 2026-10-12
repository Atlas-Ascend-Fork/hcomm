/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_LEVEL1_COMMON_H
#define CCU_LEVEL1_COMMON_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ccu_types.h"
#include "sim_common_defs.h"

namespace HcclSim {
namespace CcuSim {

    /*==================== 1. 句柄编码 ====================*/

    /**
     * @brief 原语句柄类型 tag（编码在句柄高 8 位）。
     *
     * 句柄编码（64bit）：[63:56] 类型 tag | [55:40] kernelIdx | [39:32] 保留 | [31:0] kernel 内序号。
     *
     * 设计依据：docs/CCU北向劫持-Variable族接口多进程设计.md §2.2。
     * - Variable / Address / Buffer / LocalAddr / RemoteAddr / Loop 使用 kernel 命名空间（R2 kernel 封闭），
     *   同一 kernel 录制期签发、回放期寻址，回放每次 Launch 重建 varValues 故句柄不跨 Launch 污染。
     * - Event 不入 kernel 命名空间（R2 例外）：notifyId 配对要求跨 Launch 唯一。
     */
    enum class SimCcuHandleTag : uint8_t {
        VAR = 0x01, /**< Variable（XN 寄存器抽象，64bit 标量，参与算逻/条件运算）。 */
        EVENT = 0x02, /**< Event（CKE 事件寄存器，同步专用；句柄兼作 NOTIFY 任务 notifyId 的编码原料）。 */
        ADDR = 0x03,  /**< Address（地址对象，V2 下本质 = Variable 对 / V1 下 = GSA 寄存器）。 */
        BUF = 0x04,   /**< Buffer（MS 缓冲，CCU 私有存储，伪地址建模）。 */
        LADDR = 0x05, /**< LocalAddr（本地访存三元组 = {Address + token}）。 */
        RADDR = 0x06, /**< RemoteAddr（远端访存三元组 = {Address + token}）。 */
        LOOP = 0x07,  /**< Loop/LoopGroup（硬件循环引擎对象，诊断用，桩不寻址）。 */
    };

    /**
     * @brief 组装带命名空间的原语句柄。
     * @param tag       类型 tag（高 8 位）。
     * @param kernelIdx 录制 kernel 序号（与注册表 kernelHandle 同值，bit[55:40]）。
     * @param seq       kernel 内序号（从 1 起，bit[31:0]）。
     * @return 64bit 编码句柄。
     */
    constexpr uint64_t MakeSimCcuHandle(SimCcuHandleTag tag, uint32_t kernelIdx, uint32_t seq)
    {
        return (static_cast<uint64_t>(tag) << 56U) | (static_cast<uint64_t>(kernelIdx & 0xFFFFU) << 40U)
               | static_cast<uint64_t>(seq & 0xFFFFFFFFULL);
    }

    /**
     * @brief 取句柄高 8 位的类型 tag。
     * @param handle 64bit 句柄。
     * @return tag 字节值（0x01~0x07）。
     */
    constexpr uint8_t SimCcuHandleTagOf(uint64_t handle) { return static_cast<uint8_t>((handle >> 56U) & 0xFFU); }

    /**
     * @brief 桩侧句柄 tag 校验（0 视为空句柄放行，非 0 且 tag 不符返回 false）。
     * @param handle 待校验句柄。
     * @param expect 期望的 tag。
     * @return true = 通过（0 放行或 tag 匹配）；false = tag 不符。
     */
    constexpr bool CheckHandleTag(uint64_t handle, SimCcuHandleTag expect)
    {
        return (handle == 0U) || (SimCcuHandleTagOf(handle) == static_cast<uint8_t>(expect));
    }

    /**
     * @brief 桩侧句柄 tag 校验函数族（constexpr inline，零依赖）。
     *
     * 0 视为"空句柄"放行（hccl 侧 event=0 表示无同步包裹、acq=0 表示资源桩穿刺态签发）；
     * 非 0 且 tag 不符返回 false，调用方应返回 CCU_E_PARA（fail-fast，包装层按码 throw）。
     */
    /** @brief 校验 VAR tag。 */
    constexpr bool CheckVarHandle(uint64_t handle) { return CheckHandleTag(handle, SimCcuHandleTag::VAR); }

    /** @brief 校验 EVENT tag。 */
    constexpr bool CheckEventHandle(uint64_t handle) { return CheckHandleTag(handle, SimCcuHandleTag::EVENT); }

    /** @brief 校验 ADDR tag。 */
    constexpr bool CheckAddrHandle(uint64_t handle) { return CheckHandleTag(handle, SimCcuHandleTag::ADDR); }

    /** @brief 校验 BUF tag。 */
    constexpr bool CheckBufHandle(uint64_t handle) { return CheckHandleTag(handle, SimCcuHandleTag::BUF); }

    /** @brief 校验 LADDR tag。 */
    constexpr bool CheckLocalAddrHandle(uint64_t handle) { return CheckHandleTag(handle, SimCcuHandleTag::LADDR); }

    /** @brief 校验 RADDR tag。 */
    constexpr bool CheckRemoteAddrHandle(uint64_t handle) { return CheckHandleTag(handle, SimCcuHandleTag::RADDR); }

    /*==================== 2. 变量运算种类 ====================*/

    /**
     * @brief 变量二元算术/位/移位运算种类（回放期求值使用）。
     * 对齐真身 V2 微码 operate 族：ADD=0xD / SUB=0xE / MUL=0xF / AND=0x10 / OR=0x11 /
     * XOR=0x13 / NOT=0x12 / SHL=0x14 / SHR=0x15。
     */
    enum class SimCcuVarOp : uint8_t {
        ADD = 0, /**< 加。 */
        SUB,     /**< 减。 */
        MUL,     /**< 乘。 */
        AND,     /**< 按位与。 */
        OR,      /**< 按位或。 */
        XOR,     /**< 按位异或。 */
        SHL,     /**< 左移（移位量 ≥64 结果为 0，对齐 XN 移位语义）。 */
        SHR,     /**< 右移（移位量 ≥64 结果为 0）。 */
    };

    /*==================== 3. 通用常量 ====================*/

    /** @brief 本地 CKE 事件的 notifyId 编码基址：与 DB notifyId、线程句柄空间天然隔离。 */
    constexpr uint64_t CCU_EVENT_NOTIFY_ID_BASE = 0xE000000000000000ULL;

    /** @brief CCU 缓冲(MS)伪地址基址：仅用于缓冲间重叠分析，不对应真实内存布局。 */
    constexpr uint64_t CCU_BUFFER_PSEUDO_BASE = 0x000000CC0CC000000ULL;

    /** @brief 每个 CCU 缓冲伪地址按 4K 对齐分配（步长）。 */
    constexpr uint64_t CCU_BUFFER_PSEUDO_STRIDE = 4096ULL;

    /*==================== 4. 轨迹条目类型 ====================*/

    /**
     * @brief 轨迹条目种类（原语语义的录制编码）。
     *
     * 录制-回放两段式的中间产物格式：
     * - 录制期（注册事务内，kernelFunc 执行时）：数据面 88 原语桩把语义按原始参数追加为条目；
     * - 回放期（Launch 时）：回放器逐条求值——LoadArg 绑定 taskArgs、算逻求值、
     *   if 帧抑制、while/do-while 游标循环、LoopGroup 大块抽象展开、数据原语产 task。
     */
    enum class SimCcuPrimKind : uint8_t {
        /*---- 变量/地址算逻 ----*/
        VAR_ASSIGN_IMM = 0, /**< dst=res, src=imm。 */
        VAR_ASSIGN_VAR,     /**< dst=dst, src=src。 */
        VAR_OP_VAR,         /**< dst=res, varA=lhs, varB=rhs, op。 */
        VAR_OP_IMM,         /**< dst=res, varA=lhs, imm=立即数, op。 */
        VAR_NOT,            /**< dst=res, src=src（按位取反）。 */
        ADDR_ASSIGN_IMM,    /**< dst=addr, src=imm。 */
        ADDR_ASSIGN_ADDR,   /**< dst=dst, src=src。 */
        ADDR_ASSIGN_VAR,    /**< dst=addr, src=var（基值取自变量）。 */
        ADDR_OP_VAR,        /**< dst=res, src=lhsAddr, varB=rhsVar, op。 */
        ADDR_OP_ADDR,       /**< dst=res, src=addrA, varB=addrB, op。 */
        ADDR_OP_IMM,        /**< dst=res, src=addrA, imm=立即数, op。 */
        ADDR_ADDASSIGN_VAR, /**< dst=addr, src=var（addr += var，循环步进）。 */
        /*---- 加载/存储 ----*/
        LOAD_ARG,      /**< dst=var, aux=argId（回放期从 taskArgs[argId] 绑定）。 */
        LOAD_MEM,      /**< dst=var, src=addrImm, aux=num（best-effort：值保持不变）。 */
        LOAD_MEM_VAR,  /**< dst=var, src=addrVar, aux=num（同上）。 */
        STORE_MEM,     /**< dst=addrImm, src=var, aux=num（CPU 模拟无副作用）。 */
        STORE_MEM_VAR, /**< dst=addrVar, src=var, aux=num（同上）。 */
        /*---- 组合对象 ----*/
        LOCAL_ADDR_ALLOC,  /**< dst=localAddrH, addr=addrH, token=tokenH。 */
        REMOTE_ADDR_ALLOC, /**< dst=remoteAddrH, addr=addrH, token=tokenH。 */
        /*---- 同步 ----*/
        EVENT_RECORD,        /**< event=eventH, aux=mask（本地 CKE post）。 */
        EVENT_WAIT,          /**< event=eventH, aux=mask（本地 CKE wait）。 */
        NOTIFY_RECORD,       /**< channel=chan, aux=notifyIdx, aux2=mask（远端通知记录）。 */
        NOTIFY_WAIT,         /**< channel=chan, aux=notifyIdx, aux2=mask（远端通知等待）。 */
        WRITE_VAR_NOTIFY,    /**< channel=chan, src=var, aux=remoteNotifyIdx, aux2=mask。 */
        LOCAL_NOTIFY_RECORD, /**< strIdx=tag, aux=mask（本地跨核通知记录）。 */
        LOCAL_NOTIFY_WAIT,   /**< strIdx=tag, aux=mask（本地跨核通知等待）。 */
        /*---- 本地数据 ----*/
        LOCAL_COPY_MM,     /**< dst=dstLocal, local=srcLocal, lenVar=len, event=event, aux=mask。 */
        LOCAL_COPY_MB,     /**< dst=dstBuf, local=srcLocal, lenVar=len, event=event, aux=mask。 */
        LOCAL_COPY_BM,     /**< dst=dstLocal, local=srcBuf, lenVar=len, event=event, aux=mask。 */
        LOCAL_REDUCE_MM,   /**< dst=dstLocal, local=srcLocal, lenVar=len, event=event, aux=mask, aux2=dt<<8|op。 */
        LOCAL_REDUCE_BUFS, /**< extOff=句柄数组偏移, aux=count, lenVar=len, event=event, mask=mask,
                              aux2=dt<<8|op|mask<<16。 */
        /*---- 远端数据 ----*/
        REMOTE_READ_MM,  /**< channel=chan, local=localH, remote=remoteH, lenVar=len, event=event, aux=mask。 */
        REMOTE_READ_MB,  /**< channel=chan, local=bufH, remote=remoteH, lenVar=len, event=event, aux=mask。 */
        REMOTE_READ_MR,  /**< 同上，aux2=dt<<8|op（远端读归约）。 */
        REMOTE_WRITE_MM, /**< channel=chan, local=localH, remote=remoteH, lenVar=len, event=event, aux=mask。 */
        REMOTE_WRITE_BM, /**< channel=chan, local=bufH, remote=remoteH, lenVar=len, event=event, aux=mask。 */
        REMOTE_WRITE_MR, /**< 同上，aux2=dt<<8|op（远端写归约）。 */
        /*---- 控制流 ----*/
        IF_BEGIN,    /**< lhsVar=lhs, rhs=rhs(imm或varH), aux=condType, flags.bit0=isVarCompare, strIdx=label。 */
        IF_ELSE,     /**< strIdx=label。 */
        IF_END,      /**< strIdx=label。 */
        WHILE_BEGIN, /**< lhsVar=lhs, rhs=rhs, aux=condType, flags.bit0=isVarCompare, strIdx=label, link=配对end。 */
        WHILE_END,   /**< strIdx=label, link=配对begin（回放期回跳重判）。 */
        DO_BEGIN,    /**< strIdx=label, link=配对end（体至少执行一次）。 */
        DO_END, /**< lhsVar=lhs, rhs=rhs, aux=condType, flags.bit0=isVarCompare, strIdx=label, link=配对begin。 */
        /*---- Loop/LoopGroup（大块抽象）----*/
        LOOP_BODY_ENTER,  /**< dst=loopId（体段起点边界）。 */
        LOOP_BODY_EXIT,   /**< dst=loopId（体段终点边界；体段 = (ENTER, EXIT) 之间的轨迹）。 */
        LOOP_GROUP_BEGIN, /**< dst=groupIdx（LoopGroup 占位符；展开器在回放期消费组描述符）。 */
        LOOP_GROUP_END,   /**< dst=groupIdx（组结束占位符）。 */
    };

    /**
     * @brief 轨迹条目：原语语义的原始参数（录制期固化，回放期求值）。
     *
     * 操作数槽位采用匿名 union 多视图：同一 64 位内存按 PrimKind 的语义角色命名访问。
     * 槽位语义约定（数据原语五元组统一为 channel/local/remote/lenVar/event）：
     *   槽 dst：目标句柄（赋值/运算/组合类）| channel（数据/通知原语）| lhsVar（控制流条件左值）
     *   槽 src：源操作数（赋值源 Var/Imm）| varA（二元左源）| rhs（条件右值）| local（数据原语本端）| addr（组合类
     * Address） 槽 varB：二元右操作数 | imm（立即数右源）| remote（数据原语远端）| token（组合类 token） 槽
     * lenVar：数据原语长度变量 槽 event：事件句柄（0=无同步包裹）| mask（LOCAL_REDUCE_BUFS 事件 mask）
     */
    struct SimCcuPrim {
        SimCcuPrimKind kind{SimCcuPrimKind::VAR_ASSIGN_IMM}; /**< 条目种类。 */
        uint8_t op{0};    /**< SimCcuVarOp（运算类）或 CcuConditionType（控制流类）。 */
        uint8_t flags{0}; /**< bit0: isVarCompare（控制流条件是否双变量比较）。 */
        uint8_t rsv{0};   /**< 保留。 */
        uint32_t aux{0};  /**< argId / mask / notifyIdx / count / num（按 kind 复用）。 */
        uint32_t aux2{0}; /**< dataType<<8 | reduceOp（LOCAL_REDUCE_BUFS 另含 mask<<16）。 */
        union {
            uint64_t dst{0};  /**< 槽 a：目标句柄（VAR 与 ADDR 类结果、组合类句柄）。 */
            uint64_t channel; /**< 槽 a：ChannelHandle（REMOTE 与 NOTIFY 与 WRITE_VAR_NOTIFY 类）。 */
            uint64_t lhsVar;  /**< 槽 a：控制流条件左操作数变量句柄。 */
        };
        union {
            uint64_t src{0}; /**< 槽 b：源操作数（赋值源 Var/Imm、ADDR_OP_* 左源地址）。 */
            uint64_t varA;   /**< 槽 b：二元运算左源变量句柄。 */
            uint64_t rhs;    /**< 槽 b：控制流条件右操作数（立即数或变量句柄）。 */
            uint64_t local;  /**< 槽 b：数据原语本端句柄（LocalAddr/Buffer）。 */
            uint64_t addr;   /**< 槽 b：组合类（LOCAL/REMOTE_ADDR_ALLOC）Address 句柄。 */
        };
        union {
            uint64_t varB{0}; /**< 槽 c：二元运算右操作数（变量或地址句柄）。 */
            uint64_t imm;     /**< 槽 c：立即数右源（运算/地址 imm 版）。 */
            uint64_t remote;  /**< 槽 c：数据原语远端句柄。 */
            uint64_t token;   /**< 槽 c：组合类 token 句柄。 */
        };
        union {
            uint64_t lenVar{0}; /**< 槽 d：数据原语长度变量句柄。 */
        };
        union {
            uint64_t event{0}; /**< 槽 e：事件句柄（数据原语同步包裹、EVENT_RECORD/WAIT）。 */
            uint64_t mask;     /**< 槽 e：LOCAL_REDUCE_BUFS 事件 mask。 */
        };
        uint32_t strIdx{0}; /**< label/notifyTag 在 strings 池中的下标。 */
        uint32_t extOff{0}; /**< extraBufs 偏移（LOCAL_REDUCE_BUFS 用）。 */
        uint32_t link{0}; /**< 循环条目配对索引（录制后处理填充：WHILE begin↔end、DO begin↔end）。 */
    };

    /*==================== 5. Loop/LoopGroup 描述符 ====================*/

    /**
     * @brief LoopGroup 参数来源（对齐 hccl ccu_loop.hpp 三种构造模式）。
     * - CONFIG：CcuLoopConfig/CcuLoopCfg（立即数：iterNum + addrOffset）。
     * - VAR_V1：CcuLoopGroupAddLoopFromVar（loopParam 位域打包变量，V1/A5 布局）。
     * - VAR_V2：CcuLoopGroupAddLoopFromVarV2（iterNum/addrOffset/ctxId 三个独立变量，V2/A6 布局）。
     */
    enum class SimCcuLoopParamSrc : uint8_t {
        CONFIG = 0, /**< 立即数来源。 */
        VAR_V1,     /**< 位域打包变量（V1 布局）。 */
        VAR_V2,     /**< 独立变量（V2 明文）。 */
    };

    /**
     * @brief LoopGroup 成员描述（AddLoop 时机生成）。
     *
     * 参数按来源归一：CONFIG 存立即数；VAR_V1 存位域打包变量句柄（回放期解码）；
     * VAR_V2 存三个独立变量句柄（明文，无需位域解码）。
     */
    struct SimCcuLoopMember {
        uint32_t loopIdx{0}; /**< 所属 loop（对应 SimCcuKernelEntry.loops 下标）。 */
        SimCcuLoopParamSrc src{SimCcuLoopParamSrc::CONFIG}; /**< 参数来源。 */
        /* CONFIG 来源常量 */
        uint64_t cfgIterNum{0};    /**< 迭代次数。 */
        uint64_t cfgAddrOffset{0}; /**< 每次迭代的地址步进。 */
        /* VAR 来源变量句柄（回放期从 varValues 求值） */
        uint64_t paramVar{0};      /**< VAR_V1：loopParam 位域打包（[12:0]iterNum / [44:13]gsaStride）。 */
        uint64_t iterNumVar{0};    /**< VAR_V2：迭代数变量。 */
        uint64_t addrOffsetVar{0}; /**< VAR_V2：地址步进变量。 */
        uint64_t ctxIdVar{0};      /**< VAR_V2：引擎上下文变量（记录不消费）。 */
        /* AddLoop 时刻的变量值快照（varValues 深拷贝；主/尾 group 复用同一 ccu::Loop 时取值不同） */
        std::map<uint64_t, uint64_t> varSnapshot;
    };

    /**
     * @brief LoopGroup 描述（GroupCreate 时机生成；参数含 clone 展开与偏移步进）。
     *
     * hccl 侧 LoopGroup 构造签名与来源映射：
     * - LoopGroup(paraCfg, offsetCfg, n, loops)     → CcuLoopGroupCreateFromVar    → src=VAR_V1
     * - LoopGroup(paraCfg, offsetCfg, xnOff, n, loops) → CcuLoopGroupCreateFromVarV2 → src=VAR_V2
     * - LoopGroup(cfg, n, loops)                     → CcuLoopGroupCreateCfg       → src=CONFIG
     */
    struct SimCcuLoopGroup {
        SimCcuLoopParamSrc src{SimCcuLoopParamSrc::CONFIG}; /**< 参数来源。 */
        /* CONFIG 来源常量（CcuLoopGroupConfig/Cfg 六字段） */
        uint64_t cfgCloneNum{0};        /**< 并行克隆数。 */
        uint64_t cfgCloneLoopOffset{0}; /**< 克隆起始 loop 偏移。 */
        uint64_t cfgAddrOffset{0};      /**< 克隆间地址步进。 */
        uint64_t cfgCcuBufferOffset{0}; /**< 克隆间缓冲步进。 */
        uint64_t cfgEventOffset{0};     /**< 克隆间事件步进。 */
        uint64_t cfgVarOffset{0};       /**< 克隆间变量步进（A6 专属）。 */
        /* VAR 来源变量句柄（回放期求值 → 位域解码） */
        uint64_t parallelVar{0}; /**< FromVar(V2)：parallelParam 位域打包
                                      （V1: repeatNum[61:55] / V2: repeatNum[28:19]）。 */
        uint64_t offsetVar{0}; /**< FromVar(V2)：offsetParam 位域打包（gsaStride[52:21]，V1/V2 布局相同）。 */
        uint64_t varOffsetVar{0}; /**< FromVarV2：变量偏移（记录不消费）。 */
        /* 成员列表（按 AddLoop 顺序；大块抽象以模板为单位，模板间不合并） */
        std::vector<SimCcuLoopMember> members;
        /* 轨迹区间（录制后处理填充，展开器回溯体段用） */
        uint32_t traceBeginIdx{0}; /**< LOOP_GROUP_BEGIN 轨迹索引。 */
        uint32_t traceEndIdx{0};   /**< LOOP_GROUP_END 轨迹索引。 */
    };

    /*==================== 6. Kernel 注册条目 ====================*/

    /**
     * @brief 注册的 CCU kernel 条目：注册期录制原语轨迹的容器。
     *
     * 生命周期不变量（docs/CCU北向劫持-Variable族接口多进程设计.md §2.3）：
     *  - I2 轨迹提交后不可变：读路径一律拷贝（FindKernel），禁止原地修改；
     *  - I4 句柄只以字面量嵌在轨迹中，绝不写入任何 DB 表（taskmeta 终态不含变量，R4）；
     *  - I5 进程封闭：ownerPid 记录录制进程，回放（ExecuteKernelOnLaunch）校验同进程。
     */
    struct SimCcuKernelEntry {
        uint64_t insHandle{0};     /**< 所属 CCU 实例句柄。 */
        uint32_t dieId{0};         /**< 注册时指定的 Die ID。 */
        std::string funcName;      /**< kernel 名称（允许为空，与真实语义一致）。 */
        const void* func{nullptr}; /**< kernel 执行函数（仅注册期使用，录制完成后不触碰）。 */
        const void* kernelArg{nullptr}; /**< argNum == 1 时的注册参数（仅注册期使用，Launch 期禁用）。 */
        uint32_t argNum{0};             /**< 注册参数个数，真实语义仅允许 0 或 1。 */
        uint64_t ownerPid{0};           /**< 录制进程 pid（I5 进程封闭自检）。 */
        std::vector<SimCcuPrim> trace;    /**< 录制的原语轨迹（不可变，回放期逐条消费）。 */
        std::vector<std::string> strings; /**< 轨迹字符串池（label/notifyTag）。 */
        std::vector<uint64_t> extraBufs;  /**< LOCAL_REDUCE_BUFS 的句柄数组存储。 */
        /* Loop/LoopGroup 描述符（录制期填充，回放期展开器消费） */
        std::vector<uint32_t> loopBodyEnterIdx; /**< 每个 loop 的 LOOP_BODY_ENTER 轨迹索引（下标=loopIdx）。 */
        std::vector<uint32_t> loopBodyExitIdx; /**< 每个 loop 的 LOOP_BODY_EXIT 轨迹索引。 */
        std::vector<SimCcuLoopGroup> groups;   /**< LoopGroup 描述符列表（按 GroupCreate 顺序）。 */
        /**
         * @brief 通道派生变量来源表（录制期 CcuVariableCreateByChannel 登记）。
         *
         * varHandle -> (channelHandle, varIndex)；回放期读方据此识别交换变量并懒解析
         * （hcomm CcuTransport 每 channel 预留 XN/CKE 0..3，只有被声明过的槽参与交换）。
         */
        std::map<uint64_t, std::pair<uint64_t, uint32_t>> channelVarSources;
    };

    /*==================== 7. 返回码转换 ====================*/

    /**
     * @brief 把模拟引擎的返回码映射为 CCU 错误码。
     *
     * 引擎内部（录制/注册/回放）使用 HcclVmResult，桩层需要返回 CcuResult。
     * C++ 包装层按 CcuResult throw（CCU_THROW_IF_FAILED），映射必须逐位准确。
     *
     * inline 定义在头文件中：纯函数无状态依赖，各翻译单元自行展开，
     * 不依赖 ccu_level1_common.cc 编入构建。
     *
     * @param ret 输入：HcclSim::HcclVmResult。
     * @return 对应的 CcuResult（未列出的值统一映射为 CCU_E_INTERNAL）。
     */
    inline CcuResult ToCcuResult(HcclSim::HcclVmResult ret)
    {
        switch (ret) {
            case HcclSim::HCCL_SIM_SUCCESS:
                return CcuResult::CCU_SUCCESS;
            case HcclSim::HCCL_SIM_E_PTR:
                return CcuResult::CCU_E_PTR;
            case HcclSim::HCCL_SIM_E_PARA:
                return CcuResult::CCU_E_PARA;
            case HcclSim::HCCL_SIM_E_UNAVAIL:
                return CcuResult::CCU_E_UNAVAIL;
            case HcclSim::HCCL_SIM_E_NOT_FOUND:
                return CcuResult::CCU_E_NOT_FOUND;
            default:
                return CcuResult::CCU_E_INTERNAL;
        }
    }

} // namespace CcuSim
} // namespace HcclSim

#endif // CCU_LEVEL1_COMMON_H
