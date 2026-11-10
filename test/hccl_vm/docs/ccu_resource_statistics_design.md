# CCU 资源统计特性设计方案

> 特性目标：checker 成图后，以**算子每次调用（op）× device(+die)** 为粒度统计该次调用
> 使用的全部 CCU 资源（channel、指令空间、Xn/GSA/CKE/MS 寄存器、loop、mission）；
> 成图过程中以 dfx 形式采集存储，全部成图完毕后从 dfx 提取生成 CSV 统计报告。
>
> 状态：方案设计（D1-D7 已实施，已按评审意见修订 V6）
> 涉及模块：`src/plugin/checker`
>
> V4 关键决策：
> 1. **资源采集只在单算子成图阶段进行，大图成图阶段屏蔽统计**，两条路径完全隔离；
> 2. **只按算子每次调用统计，不做 run 级/device 总量聚合**——算子处理数据量不同时
>    使用的 CCU 资源不同，run 级聚合对分析算子资源占用无意义。
>
> V5 修订：**channel 主口径由"差分"改为"翻译期引用收集"**——channel 是跨 op 持久
> 复用的资源，差分口径在 hccl_test 批次调用场景下后续 op 几乎必然为空（channelNum=0
> 大面积出现），与 per-op 统计的立论矛盾；引用口径提为一期主口径，差分降为参考（见 3.1）。
>
> V6 修订：**CSV 改为按算子分节显示，节头附带算子信息**（算子类型、算法名、数据类型、
> 元素数、输入/输出数据量、reduce 类型、rankSize），数据行精简为 14 列（见 4.2）；
> 拓扑映射修正为 `Device.id → server_id → CommConfigServer.serverKey` 主路径
> （D7 实测 deviceId 与 physical_id 口径错位，见 4.3）。

---

## 1. 背景与需求

### 1.1 需求描述

1. checker 成图完成后，统计 CCU 使用的所有资源。
2. **统计范围（V4 修订）**：以**算子每次调用**为统计单位——每次调用
   （`OpExecutionKey{commName, commHash, opIter}`）单独一份统计。原因：算子处理数据量
   不一样时，其使用的 CCU 资源也不一样（如 hccl_test 按 64B/128B/…/1G 批次调用，
   每次调用的 loop 展开数、寄存器占用各不相同）；run 级聚合统计对分析算子资源占用
   无意义，**不做 device 总量级报告**。复合用例（一次 run 调用多个算子）同样按算子区分。
3. **采集范围决策（V4）**：资源采集**只在单算子成图阶段**（`enableNewChecker` 路径，
   `CheckOneOp → GenAndCheckGraphV3`）进行；**大图成图阶段（BigGraphCheckerV3）屏蔽统计**。
   两条路径完全隔离，从根源上消除双 checker 同开时的重复计数与大图 window 内算子归属难题。
   代价：纯大图 run（`enableNewChecker=off`）无统计数据，需 WARN 声明（见 6 风险清单）。
4. 统计维度：每算子调用内按 device + ccu 统计。索引标签：`superPodId + serverId + phyDevId`
   定位 device，再按 `dieId` 定位 ccu。
5. 统计资源项：
   - **channel**：从 checker 插件的 channel 表获取，记录去重后的 channel 个数；
   - **指令空间**：CCU 指令空间为 32K 条 × 32B = 1MB，统计已生成的指令数；
   - **寄存器**：
     - A5：Xn、GSA、CKE、MS；
     - A6：Xn、CKE、MS（A6 无 GSA）；
     - **CKE 统计粒度以 ckeId 为准，不带 mask/bit 信息**；
     - **A6 引入了寄存器地址寻址，所有资源必须先将地址转换为 id，再统计**；
   - **loop**：CCU 的并发执行单元。只统计**静态展开实例数**（不含运行时迭代次数）。
     从 LoopGroup 指令计算：设 loop 模板个数为 `n`，展开 offset 为 `o`（`o ∈ [0, n-1]`），
     扩展次数为 `m`，则 loop 个数 = **`n + (n - o) * m`**；
   - **mission**：统计（V2 新增）；
   - **明确不统计**：queue、loopCtxId、jetty（jetty 由其他地方统计）、halfRTT（暂不管）。
6. 采集/报告两级结构：成图过程中以 dfx 形式采集存储统计信息；最终 checker 成图完毕后
   从 dfx 提取，生成 per-op 粒度的 CSV 报告（见第 4 节）。

### 1.2 与现有代码的对应关系（已核对）

| 需求项 | 代码依据 |
|---|---|
| deviceId 即 phyDevId | `operation_data_types.h:36`："物理设备 ID（全局唯一）" |
| 拓扑标签来源 | checker 直接从 DB 读取：`TopoMetaConfig` 表（`include/runtime_state/sim_models.h:23-26`）；checker 已有直接 Db 访问先例（`dag_graphviz_dump.cc:648` 读 Communicator 表） |
| 寄存器写入集中入口 | `AllRankParamRecorder::SetXn/SetGSA/SetCKE`（`ccu_all_rank_param_recorder_v3.cc:84/90/96`），均携带 `(deviceId, dieId)` |
| A6 地址→id 转换 | `GetXnAndTypeIdByAddr / GetXnIdByAddr / GetMSIdByAddr`（recorder .cc:279/377/406），地址段常量 .cc:259-269；checker 访问资源时已统一走地址→id 转换，统计沿用同一转换即可 |
| channel 表 | `m_allRankChannelInfo: map<DeviceId, map<dieId, map<channelId, RemoteDieInfo>>>`（`storage_manager.h:289`），由 `Loader::GetCcuChannelInfo` + `EnsureChannelInfo`（`storage_manager.cc:582`）加载 |
| 指令空间 | `CcuInstrResTab{deviceId, dieId, instrCount, instrSpace[32K][32]}`（`operation_data_types.h:117-122`），`LoadHcclVmInstrData`（`storage_manager.cc:765`）解码；32K 上限校验已存在（.cc:771） |
| mission/指令区间 | `CcuTaskParam{dieId, missionId, instStartId, instCnt}`；`CcuGraphStateV3::InitInstrInfo` 已计算 `missionStartInstrId / missionInstrCount`（`ccu_task_common_v3.cc:307-336`） |
| loop 展开公式 | A6：`ccu_task_transform_instruct_a6_v3.cc:1703` `expandTimes = (curLoopIdx < expandOffset) ? 1 : (expandCnt + 1)`，总展开 = `o + (n-o)*(m+1)` = `n + (n-o)*m`，**与需求公式一致**；A5 同构（a5_v3.cc:2056）；无嵌套（`CheckLoopGroupNotSupport` 保证） |
| LoopGroup 字段位宽 | A6：`loopInsCnt(10bit) / expandOffset(9bit) / expandCnt(9bit)`；A5：均为 7bit |
| MS 使用汇聚点 | `GenSliceFromMs(msId, len, slice)`（`ccu_task_common_v3.cc:1071`）：A5/A6 全部 MS 使用点（普通指令 + loop 展开指令，约 30+ 处调用）都在 `UpdateMSId` 偏移后以最终 msId 调用该函数 |
| dfx 现成范式 | `src/dfx/dag_graphviz_dump`：受 `enableDagGraphvizDump` 开关控制，成图内采集、结束后落盘 |

---

## 2. 总体架构

**采集只发生在单算子成图路径**；采集器随 run 复位、按算子调用分桶（`BeginOp/EndOp`），
全部算子成图完毕后一次性生成 per-op 粒度 CSV（**按算子分节显示**，每节含算子信息节头 +
该算子各 (dev,die) 数据行），不做跨 op 聚合。

```
        单算子成图阶段（采集）                     大图成图阶段            成图完毕（报告）
┌────────────────────────────────────────┐   ┌──────────────────┐   ┌──────────────────────────────┐
│ RunChecker → 遍历 op → CheckOneOp:      │   │ ProcessBigGraph   │   │ CcuResourceStatsReporter(dfx) │
│                                        │   │                  │   │  - DB 读拓扑，建               │
│  auto g = BeginOp(opKey) ◀── 采集门控    │   │  统计屏蔽         │   │    deviceId→(superPod,server) │
│  （RAII guard，析构即 EndOp 封桶；       │   │  （不 BeginOp，   │   │  - 逐 op 桶 → 分节式          │
│   须先于 LoadHcclVmSynthesisData）      │   │   钩子 no-op，    │   │    ccu_resource_stats.csv    │
│  ① Has/GetChannelInfo 引用收集          │   │   零采集/零开销）  │   │    （每算子一节：节头算子信息  │
│  ② SetXn/SetGSA/SetCKE 钩子             │   │                  │   │     + 各 (dev,die) 数据行）   │
│     （A6 先地址→id）                    │   └──────────────────┘   │  - summary 日志               │
│  ③ GenSliceFromMs 记 msId               │                          │  - 落盘 DumpV3Manager 目录    │
│  ④ TransformLoopGroupInstr key 记账     │                          └──────────────────────────────┘
│  ⑤ InitInstrInfo 记 mission/指令区间    │
│  ⑥ SetCurrentOpInfo 记算子展示信息       │
│  ~guard → EndOp() 封桶（任意退出路径）   │
│                                        │
│  CcuResourceStatsCollector (dfx, 单例)  │
│  map<OpKey, map<dev, map<die, Stats>>>  │
└────────────────────────────────────────┘
```

- **采集器**：新增 `src/plugin/checker/src/dfx/ccu_resource_stats_dfx.h/.cc`，单例。
  run 级复位点：`RunChecker` 开头（main.cc:508 附近）——**不是** `CheckOneOp` 内的
  per-op Reset（main.cc:221，那是 recorder 的复位点，恰好与本特性的 `BeginOp` 同点位）。
- **门控机制（大图隔离的关键）**：所有钩子（Set*/GenSliceFromMs/TransformLoopGroupInstr/
  InitInstrInfo）位于单算子与大图共用的代码路径，统一以"当前算子上下文"驱动：
   - `CheckOneOp` 内（main.cc:221 Reset 之后、`LoadHcclVmSynthesisData` 之前）调用
     `collector.BeginOp(opKey)`；
   - **（强制要求）`BeginOp` 必须返回 RAII guard，guard 析构即执行 `EndOp()` 封桶**。
     原因：`CheckOneOp` 在 BeginOp 之后存在多处失败提前 return
     （rank 加载/PrepareOpParam/FinalizeOpGroup/taskMeta 加载失败等，main.cc:250-301），
     裸 `EndOp()` 调用在失败路径必然遗漏，导致算子上下文悬挂、
     下一个 op 的采集数据串入上一个桶（串桶污染）。RAII 保证任意退出路径
     （含异常/CHK_RET 短路）都正确封桶；
   - 钩子入口判断：无当前算子上下文 → 直接 return（no-op）；
   - 大图路径（`ProcessBigGraph`）从不调用 `BeginOp` → 天然零采集，
     双 checker 同开也不会重复计数。
- **per-op 隔离的正确性依据**：单算子模式每 op 前 `AllRankParamRecorder::Reset()`
  （main.cc:221），寄存器状态不跨 op 泄漏——op 桶内记录的 Xn/GSA/CKE/MS 全部归属该 op，
  无"读写归属"歧义；channel 按翻译期实际引用收集（`Has/GetChannelInfo` 封装，见 3.1），
  天然归属当前 op。
- **报告器**：`RunChecker` 末尾（单算子结果汇总表之后，main.cc:594-654 附近）统一生成。
- **开关**：`CheckerSettings` 新增 `enableCcuResourceStatsDump`（`manifest.json`，默认关闭），
  模式对齐 `enableDagGraphvizDump`（`setting_manager.h:21`）；
  开关打开但 `enableNewChecker=off` 时打 WARN（无采集路径，报告为空）。

### 2.1 统计数据结构

```cpp
// dfx 内部，按 (opKey, deviceId, dieId) 一份；集合类型用于去重
struct CcuResStats {
    // ---- channel（引用口径，去重）----
    std::set<uint32_t> channelIds;       // 翻译期实际引用的 channelId（channelNum = size()，主口径）
    std::set<uint32_t> newChannelIds;    // 差分口径：本 op 新引入的 channelId（参考，不进 CSV 主列）

    // ---- 指令空间 ----
    uint32_t instrResidentNum = 0;       // CcuInstrResTab.instrCount（run 级快照，参考列，非本 op 口径）
    std::set<uint32_t> opInstrIds;       // 本 op SQE 指令区间并集（instructionNum = size()，主口径）

    // ---- 寄存器（写入时记账；A6 先地址→id；去重）----
    std::set<uint16_t> xnIds;            // xnNum = size()；另记 xnMaxId 峰值
    std::set<uint16_t> gsaIds;           // A5 only；A6 输出 NA
    std::set<uint16_t> ckeIds;           // ckeNum = size()，以 ckeId 为粒度，不带 mask
    std::set<uint16_t> msIds;            // msNum = size()

    // ---- loop（按 key 幂等记账）----
    struct LoopGroupKey {              // (missionId, instStartId, loopGroup 指令位置)
        uint16_t missionId;
        uint32_t instStartId;          // SQE 标识：同一 LoopGroup 指令可被不同 args 的 SQE 引用
        uint32_t loopGroupInstrId;
    };
    struct LoopGroupRecord {
        uint32_t n;                    // loopInsCnt（模板个数）
        uint32_t expanded;             // n + (n-o)*m（静态展开实例数）
    };
    std::map<LoopGroupKey, LoopGroupRecord> loopGroups;
    // loopGroupNum = loopGroups.size()
    // loopTemplateNum = Σ record.n
    // loopNum = Σ record.expanded

    // ---- mission（去重）----
    std::set<uint16_t> missionIds;       // missionNum = size()
};

// 算子标识
struct OpKey {  // = OpExecutionKey
    std::string commName;
    uint64_t commHash;
    uint32_t opIter;
};

// 算子展示信息（V6：CSV 节头附带；CheckOneOp 在 CheckerParam 就绪后 SetCurrentOpInfo 入桶）
struct OpInfo {
    std::string collectiveType;  // HcclCmdTypeToString(cmdType)
    std::string algName;         // OpDetailTab.algorithmName（首个非空，空则 NA）
    std::string dataType;        // HcclDataTypeToString(dataType)
    uint64_t elementCount;       // dataCount
    uint64_t inputBytes;         // CalcInputOutputSize，rank0 视角
    uint64_t outputBytes;
    std::string reduceType;      // 仅 reduce 类算子，其余 NA
    uint32_t rankSize;
};
```

聚合容器：`std::map<OpKey, std::map<DeviceId, std::map<uint32_t /*dieId*/, CcuResStats>>>`
+ `std::map<OpKey, OpInfo>`。

**不做跨 op 聚合（V4 决策）**：报告直接逐 op 桶输出，无 device 总量合并层。

> 去重语义说明：统计口径是"该算子调用成图后，哪些资源 id 被使用过"——op 桶内按 id 集合
> 去重（同一资源被多次写入/读取/清理只计一次，不按指令累加，见 3.3 通用原则 1）；
> loop 按 key 幂等记账（指令空间 + SQE 参数确定则展开结果确定，与展开次数无关，见 3.5）。
> 不同 op 之间的资源复用**不合并**——每次调用的资源占用独立呈现，这正是本特性的分析目标
> （数据量不同 → 资源占用不同）。

---

## 3. 各资源项采集细节

### 3.1 channel（channelNum）

- 数据源：`g_allRankChannelInfo[dev][die]`（`storage_manager.h:289`）。
- **主口径（V5 修订）：引用口径**——统计"该算子调用的指令翻译期实际引用的 channelId 集合"。
  原因：channel 是跨 op 持久复用的资源，差分口径（见下）在 V4 核心场景（hccl_test
  同一通信域批次调用）下后续 op 差分几乎必然为空，channelNum=0 大面积出现，数据不可用。
  **实现（已落地）**：新增统一封装 `HasChannelInfo(dev, die, channelId)`（存在性检查）与
  `GetChannelInfo(dev, die, channelId)`（取值，返回 `const RemoteDieInfo&`），
  替换现有 32 处 `g_allRankChannelInfo` 直查（a5_v3.cc 22 处、a6_v3.cc 10 处，
  已核实全部集中于两个翻译文件、模式统一；A6 `IsExistRemoteDieInfo` 保持 find 语义
  内部记账），封装内命中真实存在的 channel 时调用
  `collector.RecordChannel(dev, die, channelId)` 入当前 op 桶。引用点均在串行翻译期，
  复用 3.4 方案 A 的同一上下文机制，门控一致。
  注：图节点侧仅 `CcuNotify.channelId`（task_def_v3.h:216，同步类节点）携带 channel，
  数据搬运节点不携带 → "成图后遍历节点"路线覆盖不全，已排除。
- **参考口径（降级，可选列）：差分口径**——`BeginOp` 时对各 (dev,die) 记录 channel
  集合快照，`EndOp` 时差分得"本 op 新引入的 channelId"。仅在需要分析
  "channel 首次由哪个 op 引入"时有价值，不进 CSV 主列（dfx 内保留）。
- halfRTT 相关资源：**不统计**（评审明确暂不管）。

### 3.2 指令空间（instructionNum）

- **主口径（per-op，进 CSV）**：本 op 各 SQE 的 `[instStartId, instStartId+instCnt)` 区间
  按 (dev,die) 求并集（挂点 `CcuGraphStateV3::InitInstrInfo`，ccu_task_common_v3.cc:307，
  已有 `missionStartInstrId/missionInstrCount`），入当前 op 桶 `opInstrIds`，
  `instructionNum = opInstrIds.size()`——语义为"该算子调用在该 CCU 上使用的指令数"。
  注：`missionInstrCount` 为 uint16（ccu_task_common_v3.cc:329 强转），单 mission 指令规模
  远小于 64K，截断不影响统计准确性（评审确认，无需处理）。
- **参考口径（可选列）**：`CcuInstrResTab.instrCount` per (dev,die)
  （`LoadHcclVmInstrData`，`storage_manager.cc:765-795`，run 级完整资源快照），
  语义为"该 CCU 1M 指令空间内已生成的指令总数"，用于对照 32K 上限的占用率
  （`instrResidentNum / 32768`）；同 run 各 op 行该值相同。
- 两口径差值可暴露该 CCU 上未被本 op 引用的其他驻留指令（含其他 op 的指令段）。
- 实现优化项（非阻塞）：`opInstrIds` 用 `set<uint32_t>` 每桶每 die 最多 32K 元素，
  op 数 × die 数大时可改区间合并（interval merge）存储、报告时求总长度。

### 3.3 寄存器 Xn / GSA / CKE

**通用原则：**
1. **按 id 集合去重，不按指令累加（评审明确）**——统计口径是"该算子调用完成后，
   哪些资源 id 被使用过"。同一资源被多条指令反复操作（写、读、清理）只计 1 个：
   如 `SetXn(xn[0])` 后 `ClearX(xn[0])`，xnNum 计 1 而非 2。`std::set<id>` 结构天然满足，
   钩子只做 `insert(id)`，与指令次数无关。
   - ClearX 口径（评审已定）：A6 `ClearX` 按 id 区间逐个 `SetXn(i, 0)`（a6_v3.cc:502），
     set 去重后无双计；**凡被 ClearX 触碰的 id 一律计入使用**（不要求先被实际写入过）。
   - ClearX 异常监测（可选扩展，第一期不做）：清理从未写入过的 id 可能暗示算法编排问题
     （与 SetXn 错位、或清理冗余）。检测实现代价低（已核实 a6_v3.cc:485-510）——
     `TransformClearXInstr` 清理 `[xnId, xmId]` 区间前，检查各 id 是否已存在于
     `curXn[dev][die]`（该 map 仅由 SetXn 填充；recorder 需补一个 `HasXn` 查询接口），
     不存在即为疑似错位/冗余。注意两点：① `xnId > xmId` 时语义为清到
     `MAX_LOADX_STOREX_ID_NUM(16383)`，全量清理会命中大量未写入 id，WARN 须按指令
     聚合为计数摘要（如 "ClearX [a,b]: k ids never written"），禁止逐 id 打印；
     ② ClearX 尾部还会 `ProcessSetMask → SetCKE`，其触碰的 ckeId 按"触碰即统计"
     口径正常入 ckeIds。**现阶段只做统计、不做监测**，此项列为二期可选。
2. **写入时记账，禁止终态快照**——A6 `ClearX`、CKE wait 清位（`ClearWaitMask`）、`ClearCKE`
   会抹掉终态，快照法漏统计"用过后清除"的资源。写入时记账 + set 去重同时满足
   "不漏"与"不重"。
3. **A6 一律先地址→id 再统计**——A6 指令操作数为寄存器地址（CKE/PFE/CHANNEL/JETTY/MISSION/
   LOOP/MS 分段，recorder .cc:259-269）。**段归类在调用方完成**：翻译代码先经
   `GetXnAndTypeIdByAddr / GetXnIdByAddr` 转换并按 `CcuComponerntType` 分流后才调用
   SetXn/SetCKE（已核实：a6_v3.cc:2312-2317 按 type 分支调用），因此 `SetXn` 钩子
   天然只收到 XN 段 id，钩子内无需再做归类。
4. **loop 展开偏移后的 id 才是真实占用**——`UpdateXnId/UpdateCKEId`（a6_v3.cc:205/218，
   bit15 偏移逻辑 a6:272）之后的值入账（偏移写最终仍经 Set*，已核实无直写 map 旁路）。
5. **per-op 归属无歧义（V4）**——单算子模式每 op 前 `AllRankParamRecorder::Reset()`
   （main.cc:221），寄存器状态不跨 op 泄漏；钩子统一写入 `BeginOp` 设定的当前 op 桶，
   op 桶内记录即"该算子调用写入/占用的资源"。读取他 op 遗留值的情况在单算子模式不存在。
   跨 device 写（远端 post/搬运）以 `SetXn/SetCKE(rmtDeviceId, rmtDieId, ...)` 携带真实
   目标 (dev,die)（a6_v3.cc:2314-2317/2385/2454/2534），会正确记入对端行。

| 资源 | 挂点 | 记录内容 |
|---|---|---|
| Xn | `SetXn`（recorder .cc:84） | `xnIds.insert(id)`，另记 `xnMaxId` 峰值；A6 段归类已在调用方完成，钩子收到的即 XN 段 id（见通用原则 3） |
| GSA（仅 A5） | `SetGSA`（.cc:90） | `gsaIds.insert(id)`；A6 报告输出 `NA`（勿输出 0，避免误读为"未使用"） |
| CKE | `SetCKE`（.cc:96） | `ckeIds.insert(ckeId)`，**以 ckeId 为粒度，不带 mask/bit 信息** |

### 3.4 MS（方案供评审）

现状：recorder 无 MS 登记表，msId 不经 Set* 钩子。经代码核查，
**`GenSliceFromMs(msId, len, slice)`（`ccu_task_common_v3.cc:1071`）是 A5/A6 全部 MS
使用点的唯一汇聚函数**：普通搬运/归约指令与 loop 展开指令（约 30+ 处调用，
a5_v3.cc:834-1866、a6 同构）都在 `UpdateMSId` 偏移后以最终 msId 调用它；
A6 地址形式的 MS 也已先经 `GetMSIdByAddr`（recorder .cc:279）转成 id。
因此只需解决"该函数缺少 (dev,die) 上下文"一个问题。三个候选方案：

| 方案 | 做法 | 改动量 | 优缺点 |
|---|---|---|---|
| **A. 上下文注入（推荐）** | 复用 2 节的"当前算子上下文"门控：`BeginOp` 后，指令翻译入口（`TransformInstrQue`/`ProcessCcuNode`，已知当前 dev/die，`curCcuTask->GetDeviceId()/GetDieId()`）把 `(opKey, dev, die)` 写入 collector 当前上下文（普通成员即可，CCU 展开为单线程串行）；`GenSliceFromMs` 内部读上下文并 `RecordMs(...)` | 2 处入口 + 1 处函数体，**调用点零改动** | 覆盖所有 MS 使用路径（含 loop 合并/展开）；与采集门控共用同一上下文机制；依赖"翻译期串行"假设（已核实：ccu_graph_generator_v3 无线程，多线程仅在内存冲突检查阶段），需在代码注释中固化 |
| B. 显式传参 | `GenSliceFromMs` 增加 `(deviceId, dieId)` 参数，全部调用点传参 | 1 函数签名 + 30+ 调用点机械修改 | 无隐式状态、最健壮；改动面大，A5/A6 两文件都要动 |
| C. 图节点后遍历 | 成图后遍历 TransMem/Reduce/Batch 节点的 MS 切片（`BufferType::MS`，`offset = msId*4096` 反推 msId；loop 合并节点另有 `msIds` 集合），按节点 `TaskPosition.deviceId` + die 聚合 | 独立遍历代码 | 不侵入翻译层；但需覆盖所有节点类型，且依赖图形态（合并/并行化改写后），存在漏统风险 |

**建议选 A**（兜底可 A+C 交叉校验）。遗留确认项：A6 `GetMSIdByAddr` 的
4 CCUA bank × interleave 8 编址（recorder .cc:266-277）转换出的 msId 是 die 内全局编号
还是 bank 内编号，需与硬件规格对齐后在报告中注明口径。

### 3.5 loop（loopNum）

- **公式（已与代码核对一致）**：`loopNum = n + (n - o) * m`
  - `n = loopInsCnt`（LoopGroup 内 loop 模板个数）
  - `o = expandOffset`（前 o 个 loop 不展开，各计 1 份）
  - `m = expandCnt`（其余 n-o 个 loop 各展开 m+1 份）
  - 代码依据：a6_v3.cc:1703-1727。
- **只统计静态展开实例数，不含运行时迭代次数 loopIterCnt**（评审明确）。
- **挂点**：`TransformLoopGroupInstr` 解析完 LoopGroup 配置处
  （A6：a6_v3.cc:1654；A5：a5_v3.cc:2056）。
- **幂等记账（V4 语义）**：采集只在单算子路径、每 op 恰好展开一次，简单累加本已足够；
  key 幂等记账仍保留作为**防御机制**：
  1. 若未来在大图路径开启采集、或引入 op 重试/重复展开逻辑，计数结果仍与展开次数无关；
  2. 同 key 值冲突可作为数据异常探针。
  规则：以 `(dev, die, missionId, instStartId, loopGroupInstrId)` 为 key 写入
  `map<LoopGroupKey, LoopGroupRecord>`（见 2.1）：
   - 重复记账命中同 key 同值 → 自动去重；
   - `instStartId`（SQE 标识）必须在 key 中：同一 LoopGroup 指令可被不同 args 的 SQE 引用，
     此时 n/o/m 可能不同，属"真不同调用"，应分别计数；
   - `loopGroupInstrId` 取 LoopGroup 指令在指令空间中的绝对 id
     （`startLoopInstrId`，现场可得）；missionId/instStartId 经
     `curCcuTask`（queId → 当前 SQE 的 `CcuTaskParam`，注意 queue 内多 SQE 时取
     `ccuParamIndexs[queId]` 指向的当前 SQE）获取；
   - 防御：同 key 两次记账 value 不一致时（理论上不应发生）打 WARN 并保留首次值。
- 报告聚合：`loopGroupNum = loopGroups.size()`，`loopTemplateNum = Σ record.n`，
  `loopNum = Σ record.expanded`。
- 此层不受 loop 合并优化（`TryProcessMergedLoopA6` / `EmitMergedLoopInstrsV3`）与
  queue 并行化改写（`ApplyCcuQueueParallelizationV3`）影响；
  **禁止**从最终图节点反推（batch 合并节点会使展开数失真）。
- 无嵌套 loop（`CheckLoopGroupNotSupport` 保证），公式无需递归。

### 3.6 mission（missionNum）

- 挂点：`CcuGraphStateV3::InitInstrInfo`（ccu_task_common_v3.cc:307）或 SQE 翻译入口，
  收集 `CcuTaskParam.missionId` 入当前 op 桶的 `missionIds` 集合（按 (dev,die) 去重）。
- 附带产物：同点位可得 `missionStartInstrId/missionInstrCount`，支撑 3.2 per-op 指令区间口径。

### 3.7 明确不统计项

| 资源 | 原因 |
|---|---|
| queue | 评审明确不需要 |
| loopCtxId (xp) | 评审明确不需要 |
| jetty | 已有其他统计渠道 |
| halfRTT（wishCnt/totalCnt XN） | 暂不管 |
| PFE / `TransDep.reserve*` 预留资源 | 第一期不含，如后续需要按 3.3 地址→id 同法采集 |

---

## 4. 统计报告（CSV，V6：单份 per-op 分节式报告）

### 4.1 生成时机与位置

- 时机：`RunChecker` 末尾，全部算子成图/检查完毕后一次性生成。
- 位置：复用 `DumpV3Manager` 输出目录，单文件 `ccu_resource_stats.csv`——
  **按算子分节**（每算子调用一节：`#` 节头两行 + 14 列数据行），满足"每次算子调用单独统计"
  （hccl_test 批次调用逐次成节、复合用例多算子按节区分）。
- **不生成 run 级/device 总量聚合报告**（V4 决策：聚合数据对分析算子资源占用无意义）。
- 同时输出一条 summary 日志（风格对齐 `[CHECKER_RUN_SUMMARY]`）。

### 4.2 CSV 格式定义（V6：按算子分节显示 + 算子信息节头）

统计已按算子调用隔离，显示同样**按算子分节**：每个算子调用一节，节间空行分隔；
节头为两行 `#` 注释（算子标识 + 算子展示信息）。**节头行内字段用 `"; "` 分隔而非逗号**
（V6.1）：保证整行在 Excel 中只占第一列不被分列——节头行其余单元格为空，Excel 自动
向右溢出显示完整算子信息；同时保留 `#` 行首，pandas 等工具可按 `comment='#'` 跳过。
节内为列头 + 该算子各 (dev,die) 数据行（14 列 = 5 拓扑/设备列 + 9 资源列，
算子标识已上移节头，不再逐行重复）。

```
# ===== op[0]: commName=hccl_world_group; commHash=0; opIter=0 =====
# opInfo: type=AllReduce; alg=CcuMSAllReduceSoleMesh; dataType=INT32; elements=262144; inputBytes=1048576; outputBytes=1048576; reduceType=SUM; rankSize=4
superPodId,serverId,boardId,phyDevId,dieId,channelNum,instructionNum,loopTemplateNum,loopNum,xnNum,gsaNum,ckeNum,msNum,missionNum
0,0,0,1,1,3,235,2,128,69,31,5,256,1
0,0,0,2,1,3,235,2,128,69,31,5,256,1

# ===== op[1]: commName=hccl_world_group; commHash=0; opIter=1 =====
# opInfo: type=AllReduce; alg=CcuMSAllReduceSoleMesh; dataType=INT32; elements=1074791; ...
...
```

**节头算子信息字段**（来源 `CheckerParam` + `OpDetailTab.algorithmName`，
CheckOneOp 内在 CheckerParam 就绪后经 `Collector::SetCurrentOpInfo` 入桶）：

| 字段 | 说明 | 来源 |
|---|---|---|
| type | 算子类型 | `HcclCmdTypeToString(cmdType)` |
| alg | 算法名 | `OpDetailTab.algorithmName`（各 rank 首个非空，空则 NA） |
| dataType / elements | 数据类型 / 元素个数 | `HcclDataTypeToString(dataType)` / `dataCount` |
| inputBytes / outputBytes | 输入/输出数据量（**rank0 视角**，rank 敏感算子如 send/recv/alltoall 取 myRank=0） | `CalcInputOutputSize`（check_utils） |
| reduceType | reduce 类型；仅 ALLREDUCE/REDUCE/REDUCE_SCATTER/REDUCE_SCATTER_V 输出，其余 `NA` | `HcclReduceOpToString(reduceType)` |
| rankSize | 通信域 rank 数 | `CheckerParam.rankSize` |

**节内数据列**（14 列）：

| # | 列名 | 说明 | 来源 |
|---|---|---|---|
| 1 | superPodId | 超节点 Id | DB 拓扑映射（4.3） |
| 2 | serverId | Server Id | DB 拓扑映射 |
| 3 | boardId | Board Id（POD 形态为 board 序号；server 形态恒为 0） | DB 拓扑映射 |
| 4 | phyDevId | 物理设备 Id（= checker deviceId，全局唯一） | 采集容器 key |
| 5 | dieId | die Id（定位 CCU） | 采集容器 key |
| 6 | channelNum | 该算子调用实际引用的 channel 个数（引用口径，去重） | 3.1 |
| 7 | instructionNum | 该算子调用使用的指令数（SQE 区间并集） | 3.2 |
| 8 | loopTemplateNum | Σ n（loop 模板个数） | 3.5 |
| 9 | loopNum | Σ (n + (n-o)*m)（静态展开实例数） | 3.5 |
| 10 | xnNum | 去重后 Xn 寄存器个数 | 3.3 |
| 11 | gsaNum | 去重后 GSA 个数；A6 输出 `NA` | 3.3 |
| 12 | ckeNum | 去重后 ckeId 个数（不含 mask） | 3.3 |
| 13 | msNum | 去重后 MS 个数 | 3.4 |
| 14 | missionNum | 去重后 mission 个数 | 3.6 |

- 节排序：按 (opIter, commName, commHash)；节内行按 (superPodId, serverId, boardId, phyDevId, dieId)。
- 行覆盖：每节只输出该 op 实际涉及的 (dev,die)。单 rank 无 task 被跳过的 op（main.cc:228）不出节。
- dfx 中保留但暂不进 CSV 的扩展数据：`xnMaxId`（峰值水位）、`loopGroupNum`、
  `instrResidentNum`、channel 差分口径集合（newChannelIds）——如后续分析需要可加列，无需改采集层。

### 4.3 superPodId / serverId 映射（已实施：checker 直接读 DB）

实现步骤（`dfx/ccu_resource_stats_topo.cc`）：

1. 读 `TopoMetaConfig` 表（`sim::runtime::Db::GetOneByPred<TopoMetaConfig>`，
    checker 已有 Db 访问先例：`dag_graphviz_dump.cc:648`）；表项存 topo yaml 文件名，
    `"ranktable"` 表示 ranktable.json 模式。
2. 调 `sim::runtime::ParseCommConfigData(fileName, nullptr, commConfig)`
   （runtime_state 导出，checker 已链接；yaml 与 ranktable.json 双模式统一处理），
   得到 `commConfig.servers[]`（含 podId/serverId/serverKey/phyIds）。
3. **主路径（D7 实测修正）**：查 `Device` 表建立
   `deviceId(=Device.id) → server_id → CommConfigServer.serverKey → (podId, serverId)` 映射。
   **陷阱**：checker 的 deviceId（Device.id，实测 1-based）与拓扑 phyIds
   （Device.physical_id，0-based）口径错位，直接用 deviceId 匹配 phyIds 会导致
   标签错位/缺失（D7 第一轮实测 dev4 → -1 即此因）。
4. 兜底路径：`phyIds` 直接匹配（deviceId == phyDevId 的部署场景），emplace 不覆盖主路径。
5. 任一环节失败（表项缺失/文件不可达/解析失败）→ 空表 + WARN，两列输出 `-1`，
   不阻塞报告生成。

---

## 5. 可行性结论

**方案可行**：

1. **采集只在单算子成图阶段**（V4 决策）：该路径每 op 独立执行
   `CheckOneOp → GenAndCheckGraphV3`，且每 op 前 recorder Reset（main.cc:221），
   per-op 统计边界天然干净——寄存器无跨 op 泄漏、channel 按翻译期引用收集归属（V5），
   "算子调用能否分开统计"的答案是**能，且两种典型场景都覆盖**：
   - hccl_test 批次调用：每次调用 opIter 递增 → 独立 OpExecutionKey → 独立统计桶；
   - 复合用例多算子：不同算子必然不同 key（commName/commHash/opIter 三元组唯一）→
     独立统计桶。区分依据是输入数据模型的一等公民身份，不依赖启发式切分。
2. **大图路径不采集**，通过"BeginOp 上下文门控"实现零采集零开销，双 checker 同开
   也无重复计数；大图 window 内多算子共享状态、TaskPosition 归属等难题全部消解。
3. 寄存器写入有集中入口（SetXn/SetGSA/SetCKE）；A6 地址→id 转换 checker 已统一实现，
   统计沿用即可；MS 有唯一汇聚点 `GenSliceFromMs`（3.4 方案 A 零调用点改动）。
4. loop 公式与代码实现一致（a6_v3.cc:1703），挂点在翻译层，不受合并/并行化改写影响；
   key 幂等记账作为防御机制保证计数与展开次数无关。
5. 拓扑标签经 DB 直读可达；dfx 采集 + 结束出报告有现成范式（dag_graphviz_dump）。
6. **已知代价**：纯大图 run（`enableNewChecker=off` 且 `enableBigGraphChecker=on`）
   无统计数据——统计特性显式依赖单算子 checker 开启，开关冲突时 WARN（见 6 风险 #4）。

---

## 6. 风险与遗留问题清单（V5 修订）

| # | 风险/缺口 | 等级 | 对策 |
|---|---|---|---|
| 1 | 拓扑解析工具库（src/cmd / src/runtime_state）checker 未链接；且 `TopoMetaConfig` 仅存 yaml 文件名/`"ranktable"` 标记，checker 进程运行环境下拓扑文件的绝对路径解析方式未验证 | 中 | 下沉公共库或 checker 内等价实现；实施前验证文件路径可达性；兜底输出 -1 |
| 2 | `ccu_resource_base_addr_` 当前为 mock 硬编码（storage_manager.cc:757-763，仅 2 个 die 槽位），影响 A6 地址→id 转换；dieId≥2 时 `GetXnAndTypeIdByAddr/GetMSIdByAddr` 直接报错，A6 多 die 统计会失败而非降级 | 中 | 统计复用 checker 既有转换路径（与检查逻辑同口径，不引入新误差）；转换失败时记 UNKNOWN 桶 + WARN 不中断；base addr 正式化属存量问题，随其修复自动生效 |
| 3 | MS 统计方案待评审拍板（3.4 A/B/C）；A6 msId 的 bank/interleave 编号口径需与硬件规格对齐 | 中 | 推荐方案 A；口径确认后在报告注明 |
| 4 | 纯大图 run 无统计数据（V4 采集范围决策的代价） | 中 | 统计开关打开但 `enableNewChecker=off` 时打 WARN；文档/报告头注明依赖；如后续需要大图统计，二期按 TaskPosition.operatorId 归属扩展（key 幂等记账已为其兜底防双计） |
| 5 | instructionNum 主口径为 per-op SQE 区间并集，instrResidentNum（run 级快照）为参考值，两者语义需在使用说明中写清 | 低 | 3.2/4.2 已定义；报告头注释口径 |
| 6 | 检查失败提前退出时是否输出部分报告 | 低 | 建议仍输出（数据为已采集部分），日志标记 partial；失败 op 桶保留并在 summary 列出 |
| 7 | Set/GenSliceFromMs 为高频路径，采集钩子有性能开销 | 低 | set 插入 + 开关默认关闭 + 无上下文时首行短路 return；采集期上下文为普通成员无锁 |
| 8 | 复位/门控点位易错：collector run 级复位须挂 main.cc:508 附近；`BeginOp` 须挂 CheckOneOp 内 recorder Reset（main.cc:221）之后（参考口径 newChannelIds 的差分快照还须先于 `LoadHcclVmSynthesisData`，main.cc:244） | **高** | **`BeginOp` 必须返回 RAII guard（析构即 EndOp）**：CheckOneOp 在 BeginOp 后有多处失败提前 return（main.cc:250-301），裸 EndOp 在失败路径必然遗漏 → 上下文悬挂、下个 op 数据串桶。RAII 覆盖全部退出路径（见 2 节门控机制）；单测须覆盖"op 中途失败 → 下一 op 桶不被污染"场景 |
| 9 | 双 checker 同 run 叠加（main.cc:536-560）：V4 大图路径零采集，架构上已消除双计；剩余风险为未来有人在大图路径误加 BeginOp | 低 | 3.5 key 幂等记账兜底；大图路径代码注释声明"禁止采集"；同 key 值不一致时 WARN 并保留首次值 |
| 10 | 多 die 场景：A5 设备 dieId 取值范围与 A6 不同，报告行需按实际出现过的 (dev,die) 生成 | 低 | 以采集容器实际 key 为准 |
| 11 | op 失败重跑/重试类场景（若未来引入）会导致同 op 桶多次写入 | 低 | BeginOp 时若桶已存在则 WARN 并选择覆盖或合并（实现时定） |
| 12 | ~~channel 差分口径的边界~~（V5 已解决）：差分口径在批次调用场景后续 op 恒为空，与 per-op 立论矛盾 | 已关闭 | 3.1 主口径改为翻译期引用收集（`Has/GetChannelInfo` 封装替换 32 处直查）；差分降为参考口径（newChannelIds，不进 CSV 主列） |
| 13 | channel 引用收集需替换 32 处 `g_allRankChannelInfo` 直查（a5_v3.cc 22 处、a6_v3.cc 10 处），机械修改但点位多，漏改会少计 | 低 | 引用模式统一（`g_allRankChannelInfo[dev][die]...`），可用 grep 校验清零直查点；封装函数强制走记录路径；单测对照 golden channel 集合 |

---

## 7. 实施拆分（建议）

1. **M1 dfx 采集框架**：`ccu_resource_stats_dfx`（OpKey 分桶容器 + run 级 Reset 挂接 +
   `BeginOp` 返回 RAII guard（析构即 EndOp，覆盖 CheckOneOp 全部失败提前返回路径）+
   开关 + 开关冲突 WARN）。
2. **M2 寄存器采集**：SetXn/SetGSA/SetCKE 钩子（无上下文短路；A6 地址→id 归类后入账）。
3. **M3 MS 采集**：按评审选定方案（建议 3.4-A 上下文注入，与 M1 门控共用上下文）落 `RecordMs`。
4. **M4 loop/mission/指令区间采集**：TransformLoopGroupInstr 处 key 幂等记账
   （含同 key 值冲突防御）、InitInstrInfo 收集 mission/per-op 指令区间。
5. **M5 channel 采集**：新增 `HasChannelInfo/GetChannelInfo` 统一封装（内嵌 `RecordChannel`
   引用收集），替换 a5/a6 翻译文件 32 处直查；BeginOp/EndOp 差分逻辑保留为参考口径（newChannelIds）。
6. **M6 拓扑映射 + CSV 报告器**：DB 读 TopoMetaConfig、`ParseCommConfigData` 解析、
   Device.server_id→serverKey 反查，逐 op 桶生成单份分节式 `ccu_resource_stats.csv`
   （节头算子信息 + 13 数据列，见 4.2）+ summary 日志。
7. **M7 测试**：loop 公式单测（o=0、o=n-1、m=0 边界）、多 op 分桶/去重场景
   （hccl_test 批次模拟，**重点验证批次调用下 channelNum 引用口径非零且逐 op 稳定**）、
   channel 引用收集对照 golden 集合（校验 32 处直查替换无遗漏）、
   **op 中途失败 → RAII 封桶 → 下一 op 桶不被污染**、
   A5/A6 双版本 golden CSV、双 checker 同开零采集验证。
