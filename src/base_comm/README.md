# base_comm 模块代码解析

## 功能描述

base_comm 是 hcomm 仓的基础通信层（L3，最底层）：资源管理 + 通信原语执行。按仓根 AGENTS.md 的
分层依赖规则，base_comm 不可反向依赖 coll_communicator_mgr / coll_comm_ops。

核心能力：

- 通信资源管理：endpoint、channel、socket、注册内存、CCU 资源等（resources/）；
- 通信原语执行：host 侧 C 适配接口与 AICPU 侧原语（primitives/）；
- 配置与基础设施：配置管理（config_mgr/）、DFX 异常上报（dfx/）、动态库符号加载等公共组件（common/）；
- 资源树管理：进程级/设备级资源统一收敛到以 BaseCommMgr 为根的资源树，治理进程退出期静态对象
  乱序析构——详见"资源树管理"节，改动树结构、增删节点、调整成员声明顺序前必须先读该节。

## 目录描述

```text
base_comm/
├── common/                          公共组件（TpMgr / EidInfoMgr / HccpTlvHdcMgr / orion 适配 / 动态库符号加载）
├── config_mgr/                      配置管理（ConfigMgr / CommConfiger 适配）
├── dfx/                             DFX（异常处理与上报）
├── primitives/                      通信原语（api_c_adpt C 接口适配 / aicpu / dfx）
├── resources/                       通信资源（ccu / comm_engine_res / endpoint_pairs / endpoints /
│                                      flush / hccp / reged_mems / southbound_adpt）
├── base_comm_mgr.{h,cc}             资源树树根 BaseCommMgr
├── base_comm_node_accessors.{h,cc}  树窄访问器（调用点唯一入口）
├── base_comm_res_per_dev.{h,cc}     设备子树（按 devPhyId 每卡一棵）
├── handle_registry_procs.{h,cc}     Channel / Thread 句柄注册表节点
├── legacy950_res_mgr_proc.{h,cc}    legacy 单例锚定节点
└── README.md / README_en.md         本文档
```

（文件间关系表与组件交互图待补充）

## 流程描述

- 创建：进程级节点随 `BaseCommMgr` 首次 `GetInstance()` 构造期一次建齐（R6）；
  设备子树按 devPhyId 首访懒建，经窄访问器返回引用（R7/R9）。
- 销毁：显式 `DeInit(ExitProfile)` 走 9 步动作链（R4，exit_path ST 可观测校验）；
  析构只释放内存，`~BaseCommMgr` 兜底重放 DeInit 链。

## 数据描述

- 树形分层：进程级 = 全进程一份或键不含设备维度；设备级 = 按 devPhyId 每卡一份
  （判定规则见「树的形态」节）。
- 设备子树存取：`BaseCommResPerDevMgr` 维持现役表 + 退役区 + reserve 兜底（R9），
  稳态读无锁、首建持锁。

## 接口描述

本模块对上层暴露的统一入口为窄访问器（`base_comm_node_accessors.h`，用法见 R7），
按键型分三组：

- 进程级无键：`GetConfigMgr` / `GetCommConfiger` / `GetDlUrmaFunction` /
  `GetAicpuCommEngineBinProc` / `GetEndpointHandleMgr` / `GetSharedJettyMgr` /
  `GetProcRegedMemMgrCache` / `GetEndpointProcMgr` / `GetServerSocketMgr` /
  `GetChannelHandleRegistryProc` / `GetThreadHandleRegistryProc` / `GetDpuNotifyManager` /
  `GetProcRegedMemMgrCacheHolder`（shared_ptr 非 owning 别名，见 R7）；
- 设备级 phyId 键：`GetEidInfoMgr` / `GetEndpointCtxMgr` / `GetSocketMgr` /
  `GetGlobalNetDevMgr` / `GetTpMgr` / `GetHccpTlvHdcMgr`；
- 设备级 logicId 键：`GetCcuKernelMgr` / `GetCcuInstanceMgr` / `GetCcuVarEventResMgr` /
  `GetSocketProcess`。

越界/失败回落口径见 R9；设备级访问器入参键型不一致是历史沿袭，禁止混用。

## 使用限制

- 构造期禁止重入窄访问器（死锁红线，R2）；
- 业务代码只准走窄访问器，禁止直接写树访问链、禁止新增文件级全局（R1/R7）；
- `EXIT_PROFILE_RESET` 下跳过一切触设备动作，endpoint/channel 表残留的设备侧归还
  由析构链兜底、不保证归还（已知限制见 R5）；
- 越界回落语义为基线兼容行为，三种口径并存（R9），新增访问器须先登记口径；
- 收编单例入树或改调用点前，必须反向排查其析构链是否触树（R10）。

## 资源树管理

本目录的资源由一棵**进程级资源树**统一管理。改动树结构、增删节点、调整成员声明顺序前，
必须先读完本节；不符合规则的修改不得合入。

设计原则：单根（R1）/ 构造期禁重入（R2）/ 所有权即生命周期（R4）/ 粒度分层（树形态）/
声明序=析构契约（R3）/ 最小侵入 / 二进制边界（R8）/ 面向演进（占位壳预留落点，二期填内容不改树形）。

### 树的形态

```text
BaseCommMgr（进程级树根，base_comm 唯一新增单例；存量函数级 static 工厂见 R1 例外清单，由 R8 锚定）
├── [基础设施] isDeInited_ / lastDeinitWasReset_                   ← 声明最前 = 全树最后死
├── Legacy950ResMgrProc    legacy950 单例锚定节点（libhccl_v2.so，只锚定不入树）
├── ConfigMgrProc          配置（ConfigMgr 值成员；CommConfiger 转发）
├── DlsymProc              动态库符号表（DlUrmaFunction）
├── DfxProc                异常上报 / profiling
├── EndpointProcMgr        进程级网络资源：网卡/连接表 + ServerSocketMgr
├── NotifyMgrProc          notify 管理（DpuNotifyManager，进程级语义：ID 空间全进程共享）
├── CommEngineBinMgrProc   进程级 bin 族（Aicpu 实体 + Aiv/Dpu 占位）
├── BaseCommResPerDevMgr   设备子树存取器（现役表 + 退役区 + 拒建兜底）
│     └── BaseCommResPerDev（每张物理卡一棵，首访懒建）
│           ├── NetDevProc          网卡设备级实例
│           ├── ServerSocketMgrProc SocketMgr
│           ├── SocketProcess       Socket 清理（析构经 GetSocketMgr 还 socket）
│           ├── CommEngineMgrProc   引擎族：ThreadMgr / MemMgr / HdcProc / CcuInstanceProc
│           └── EndpointCtxMgr      EndpointCtx 去重缓存
├── HandleRegistryMgrProc  Channel / Thread 句柄注册表（残留 channel 析构触设备子树，先死）
└── EndpointMgr            endpoint 族（句柄表 + SharedJetty + RegedMem 缓存 + EidInfo）← 全树最先死
```

层级判定：**进程级** = 全进程一份或键不含设备维度；**设备级** = 按 devPhyId 每卡一份。
设备无关的进程级资源（如 bin 句柄）不得放进设备子树。

### 核心规则

#### R1 单根，不新增全局变量

base_comm 内不得再新增 Meyers 单例 / 函数级 static 对象 / 文件级全局变量。
需要全局语义的资源，挂到树对应层级的节点上做值成员，经窄访问器（R7）暴露。

例外（须经评审并在 README 登记）：

- 跨二进制（非 hcomm_base_obj 编译单元）搬不动的 legacy 单例（libhccl_v2.so 的
  `Hccl::` / `hccl::CommConfiger`）——只锚定不入树（R8）；
- ACL 回调注册标志（`g_deviceRefreshCallbackRegistered` 等）——生命周期归 ACL 管控，不受树管控；
- reserve 兜底静态实例（R9）；
- base_comm 自有存量单例（开源开放 CCU 族：`CcuPfeCfgMgr`、`hcomm::CcuComponent`、
  `hcomm::CcuResBatchAllocator`、`hcomm::CcuResSpecifications`，key=userDevId 与设备子树
  phyId 键不一致/当前混跑不使用）——维持锚定（R8）暂不入树，二期评估收编；
- AICPU 专属组件（`ExceptionHandle` / `ExceptionCallbackMgr`，仅随 ccl_kernel 编译）——
  host 进程无此对象，不进树、不锚定；
- legacy 平台底座适配（`hrtTryGetDeviceIndexByPhyId` / `hrtTryGetDevicePhyIdByIndex`，
  src/legacy/ascend910）——锚定（R8）与窄访问器预期回落所需的 phyId↔logicId 双向
  探测性转换 quiet 变体，为存量 static `GetDeviceIndexByPhyId(...)` 与同源拆出的
  `GetDevicePhyIdByIndex(...)` 的薄包装、无新实现逻辑；
  VM/缺卡场景转换失败是预期路径，非 quiet 版在 legacy 内部 return 前刷 ERROR、
  外层封装拦不住，故必须以 quiet 变体新增接口；符号沿用既有 .map 导出机制。
  属仓根 AGENTS.md「禁止在 src/legacy/ 新增接口」约束的登记例外，
  二期推动 legacy 提供正式 quiet 变体后回收；
- host 侧存量 static（均编入 hcomm_base_obj，析构不经窄访问器、不触树，无 R10 UB 路径，
  维持现状不锚定）：
  - `FlushManager`（resources/flush/flush_manager.cc）——首触方在上层 coll_comm.cc，
    析构 DestroyAll 销毁 QP/CQ/MR 为 master 既有行为，依赖的 `RdmaHandleManager` 已锚定（R8）；
    二期评估上提为树根直属进程级节点；
  - `Hccl::DfxDlProfFunction`（dfx/dfx_dlprof_function.cc）——析构仅 dlclose；
  - `EnvPlfDebugConfig`（common/debug/plf_debug_config.cc）——环境变量解析 + 原子值，
    析构平凡；
  - NicPlugin 注册表簇（primitives/api_c_adpt/nic_plugin/nic_plugin_manager.cc 的
    `LoadOnce` / `LoadedPlugins` / `ProtocolPlugins`）——析构卸载插件（dlclose 链）；
  - `AicpuTsRoceEndpoint::NetDevMapMutex/GetNetDevMap`（resources/endpoints/
    aicpu_ts_roce_endpoint.cc）——host 侧 netdev 引用表；
  - `AicpuTsRoceServerSocketContext::ListenSocketMapMutex/GetServerSocketMap`
    （resources/endpoints/server_socket_context/aicpu_ts_roce_server_socket_context.cc）——
    host 侧 listen socket 表；
- 树亡标志（base_comm_mgr.cc 函数级 static 原子，经 `hcomm::IsResourceTreeTeardownFinished()`
  查询）——平凡可析构、树死后读取安全，供锚定单例（晚于树死）析构链判活跳过触树归还（R10）。

#### R2 构造期禁止经窄访问器重入（死锁红线）

设备子树首访懒建由 `FindOrCreate()` 慢路径全程持 `BaseCommResPerDevMgr` 内部锁（非递归互斥锁）完成，
节点构造函数与 `AnchorDeviceSingletons()` 均在持锁状态下运行。此期间调用任何设备级
窄访问器（`hcomm::GetSocketMgr` 等）即同线程二次加锁 → 死锁；且只在首访建子树路径偶发，
极难排查。

- 锚定只针对 legacy 单例：直接调其 `GetInstance()`，一律不经窄访问器；
- 树上的值成员随构造建齐，不需要也不允许"锚定"；构造期引用兄弟节点直取成员；
- 构造路径上不得调用窄访问器或任何会回调 `BaseCommMgr::GetInstance()` /
  `GetBaseCommResPerDev()` 的入口（半构造递归 / 持锁重入死锁）。

#### R3 成员声明顺序 = 析构契约（先声明的后死）

C++ 成员按声明逆序析构。每个树节点内的成员声明顺序就是析构顺序契约：

- 声明一个成员前，先回答："谁析构时要回调/归还到我？" 被依赖者必须先声明（后死）。
- 基础设施（锁、DeInit 标志）声明在一切使用者之前（最后死）——析构期回调链要持锁。
- **改动任何声明顺序 = 改动析构契约，必须单独评审**，并在本 README 登记理由。

进程级契约（先死 → 后死）：
`EndpointMgr → HandleRegistryMgrProc → 设备子树 → CommEngineBinMgrProc → NotifyMgrProc → EndpointProcMgr → Dfx/Dlsym/Config → Legacy950ResMgrProc → 锁与标志`

设备子树契约（先死 → 后死）：
`EndpointCtxMgr → CommEngineMgrProc（停 Thread → Mem → HDC → CCU）→ SocketProcess → ServerSocketMgrProc → NetDevProc`

EndpointMgr 内部契约（先死 → 后死）：
`EndpointHandleMgr → ProcRegedMemMgrCache`——endpoint 由句柄表独占持有，其析构经非 owning 别名
cacheAlias_ 解引用 cache（R7 别名规则）；EndpointMgr 其余成员间无约束。

登记（声明序变更，评审 HIGH 闭环）：

- `handleRegistryMgrProc_` 后移至 `perDevResMgr_` 之后——RESET/退出兜底路径下残留 channel 随句柄表
  析构，其析构经窄访问器还 socket/notify，必须先于设备子树死（否则触已析构对象）；
- `BaseCommResPerDevMgr` 内 `mtx_`/`deinited_` 前移至 `reserve_` 之前——退役区析构期回落 reserve_
  的慢路径须持锁读闸，锁与闸须最晚死。
- `ChannelHandleRegistryProc` 内 `channelMapMtx_` 前移至三表之前——三表析构期（channel 析构经访问器
  加锁）锁须最晚死；原顺序锁先于表死，表析构期加锁命中已析构的 std::mutex（UB）。
- `ThreadHandleRegistryProc` 内 `engineThreadMapMtx_` 前移至两表之前——防御性：map 析构期释放
  `shared_ptr<Thread>` 时若 `~Thread` 回调访问器加锁，锁不得先死（当前派生类析构只做 DeInitImpl 不回调）。
- `reserve_` 维持声明在 `active_`/`published_`/`retired_` 之前（评审建议后移，已驳回，勿再提）：
  退役区析构期回落须拿到存活 reserve_；reserve_ 从不 Init、子树默认析构，其析构链不重入
  FindOrCreate，评审假设的重入路径不存在；后移反而使 retired_ 析构期回落命中已析构对象（真 UAF）。

#### R4 生命周期双路径：DeInit 管动作，析构管内存

- `DeInit(ExitProfile)`：显式动作路径（停线程、清表、归还设备资源），按契约顺序执行，**幂等**
  （树根用原子 exchange 保证首次生效；节点级 DeInit 自带幂等）。
- 析构函数：只回收内存，不写动作逻辑；动作逻辑的兜底由 `~BaseCommMgr` 先走一遍 DeInit 链承担。
- DeInit 完成后设备子树从现役表摘入退役区：动作链执行期间树必须完整可解析（析构链回调要走访问器）。
- DeInit 与并发首访：DeInit 起始即置拒建闸（`MarkDeinited`），慢路径持锁复查、拒建回落 reserve；
  动作链遍历走锁内快照（`SnapshotActive`），与并发插入互斥。
- DeInit 动作链的顺序依据（与实现 step[N/9] 日志标记一一对应，新增/调整动作时必须维持）：
  1. step[1/9] 停 Thread 最先——活线程会回调正在死亡的对象（当前为占位：ThreadMgrProc 无实体、
     StopThreads 零动作，线程实际随句柄表节点析构停止，声明序已保证先于设备子树死亡；
     显式停止为二期工作项）；
  2. step[2/9] 清 Endpoint——endpoint 析构链回调设备侧 EndpointCtxMgr、
     归还网卡上下文与注册内存引用，须在引擎族收尾前；
  3. step[3/9] 清残留 Channel——channel 析构经窄访问器归还 socket/notify，须在引擎族收尾前；
  4. step[4/9] 逐设备清 SocketProcess 表与 EndpointCtxMgr 残留——还 socket/强释 ctx 触 SocketMgr/CCU，
     须在引擎族收尾前；endpoint 析构已回调 Release，此处清剩余；
  5. step[5/9] 引擎族收尾——ctx 回调归还 CCU 资源时 CCU 管理器必须存活；
  6. step[6/9] 卸 bin——引擎族收尾后不再引用 bin（当前为占位：无卸载逻辑，
     bin 句柄随进程退出由 OS 回收）；
  7. step[7/9] 进程级网络资源——还网卡上下文/关 listen socket 时 EndpointProcMgr 必须存活；
  8. step[8/9] 基础设施与设备状态回调——纯 host 动作，两 profile 相同（channel 表清理已随第 3 步完成）；
     refresh 回调按名注销并复位标志（复位是退出语义，允许重新注册；
     reset 回调与 ACL 按名配对、随进程退出失效，不注销）；
  9. step[9/9] 摘表移交退役区最后做——动作链期间正式表必须可解析（析构链回调经窄访问器）。

#### R5 双 profile 语义（exit_profile.h）

- `EXIT_PROFILE_GRACEFUL`（默认）：完整动作链，含触碰设备的动作；进程退出兜底固定走它。
- `EXIT_PROFILE_RESET`：设备可能已不可用，跳过一切触碰设备的动作，仅 host 侧清理；
  只由显式调用触发，动作顺序与 graceful 同序（不做的动作显式跳过、不重排）。
- 新增"动作"时必须回答它属于哪个 profile：触设备的动作 RESET 下必须跳过。
- 命名辨析：`OnDeviceResetPre` 设备热复位回调是按设备的局部清理，不触发整树 DeInit，不对接本 profile。
- 已知限制：RESET 跳过 endpoint 表与 channel 表的设备侧归还；此后进程退出若句柄表仍有残留，
  成员析构仍会触设备（`~BaseCommMgr` 打 WARNING 暴露）。RESET 下引擎族（HDC/CCU 实例）收尾同样
  跳过（节点打 WARNING），但析构不被抑制：`~CcuInstanceMgr` 仍会清实例表、触发
  `~CcuInstance` 的设备归还链——RESET"跳过触设备"的承诺在析构期被穿透。当前 RESET
  无生产调用方（仅 exit_path ST 显式触发），登记为已知限制；接入真实调用方前须先落地
  析构抑制（置标志使析构故意泄漏设备侧资源，host 内存随进程退出回收）。
  RESET 下进程级网络资源同样不在动作链清理：`EndpointProcMgr` 的 netDevCtx/serverSocket
  表与各设备 `NetDevProc` 的 `GlobalNetDevMgr` 均推迟到析构，且 `~GlobalNetDevMgr` 会调
  `HcclNetDeInit` 触设备——RESET「跳过触设备」的承诺在该路径同样被析构期穿透。

#### R6 构造期建齐，getter 不留无锁懒建

- 节点的子节点在**构造函数中一次建齐**（含 reserve 兜底子树），getter 直接返回成员引用。
  无锁的 `if (nullptr) reset(new)` 懒建是并发首访数据竞争，禁止新增，无例外。
- 构造期禁止重入窄访问器（死锁红线，见 R2）。

#### R7 调用点一律走窄访问器

树访问链很长且超列宽，业务代码一律包含 `base_comm_node_accessors.h`，用
`hcomm::GetXxxMgr(...)` 短自由函数（返回引用），**不直接写树访问链**。
新增节点时同步新增对应窄访问器。
本头仅供 host 侧编译单元包含（ccl_kernel/AICPU 侧无资源树）；访问器返回树节点
持有的实例引用，对象身份与生命周期随树，不新增全局变量、不做注入。
存量 `shared_ptr` 接口用空 deleter 非 owning 别名兼容（如 `GetProcRegedMemMgrCacheHolder()`）：
别名只借用地址、计数不管真身生死，生命周期由树的声明序保证；别名必须死在真身之前
（持有方声明序先于真身）。此类别名属一期适配器，二期可去除。
节点类当前以 public 可变引用 getter 暴露内部容器与锁（如 `ChannelMap()`/
`ChannelMapMtx()`）：这是迁移期过渡形态——基线文件级 static 仅本 TU 可见，收编后
编译期无法再约束锁纪律，「单锁护三表」依赖调用方自觉。封装收敛（命名操作 + 类内
持锁，删除可变 getter）为二期专项工作项；存量复合操作（一次持锁多表联动）在收敛时
必须保持同一把锁的原子粒度。
文件级局部别名（如 thread.cc 的 `ThreadRegistry()`）仅允许作为列宽缩写存在，
函数体必须直调窄访问器，不得直写树访问链。

#### R8 legacy 单例只锚定不入树

libhccl_v2.so 内的单例跨二进制搬不动，只锚定不入树；锚定只做生命周期控制，不执行业务动作。
两级锚定的保证强度不同：

- 进程级：`Legacy950ResMgrProc` 构造期首触，在树根构造期内完成注册——锁死晚于全树死亡；
- 设备级：依赖清单唯一（`AnchorDeviceSingletons`，构造平凡、无副作用）。设备子树懒建的
  逐槽首触晚于树根构造，单靠它锁不死（静态析构逆序反而先死）；故 `Legacy950ResMgrProc`
  构造期以备份槽位（MAX_MODULE_DEVICE_NUM）调用一次完成预热，static 数组在树根构造期内
  完成注册 = 锁死晚于全树死亡；设备子树构造期按本槽再调（UT/独立构造场景生效，正常路径
  no-op）。

新增 legacy 依赖时在对应锚定点补一行首触（设备级单例补进 `AnchorDeviceSingletons` 即可）。

#### R9 越界回落兜底

设备级访问器保持"越界/转换失败回落兜底"语义；入参键型与原 GetInstance 一致：
CCU 三件套（CcuKernelMgr/CcuInstanceMgr/CcuVarEventResMgr）为 deviceLogicId，
其余为 devPhyId：

- phyId 键访问器：0~65 均为合法设备（65 为备用卡槽位，取对应设备子树真实实例）；
  越界回落口径与基线逐一对齐，三种并存（勿为"统一"而改行为）：
  - `GetSocketMgr`：基线数组无备份槽，越界（含 65）回落 0 号设备 + WARNING（socket_mgr.cc）；
  - `GetEidInfoMgr`/`GetTpMgr`/`GetHccpTlvHdcMgr`：≥65 钳位 65 + WARNING（基线同款，
    65 备用卡也告警，eid_info_mgr.cc）；
  - `GetGlobalNetDevMgr`/`GetEndpointCtxMgr`：>65 钳位 65 + WARNING，65 静默合法
    （基线 GetDeviceResMgr 对显式 65 逐次告警；本 PR 有意改静默——65 为合法备用槽，
    合法使用逐次告警是噪音，行为变更登记于此）；
- logicId 键（CCU 三件套）：越界 logicId 回落进程级 reserve 实例（类静态）；
  reserve 进程级唯一，多个非法 id 共享时不再按设备隔离（异常兜底，正常路径不触达）。
  id 成员改由 owner 在 Init 期设定（CcuDrvHandle::Init / CcuInstance::Init），访问器不再
  回写——共享 reserve 的并发回写是数据竞争（评审闭环）；reserve 从不 Init，id 保持默认 -1。
  「不 Init」为约定而非强制：非法 id 若触发 Init（如 CcuDrvHandle::Init 经
  转换失败回落），reserve 析构将执行 hrtFree/Deinit 触设备动作——与基线 65 号备份槽
  风险等同（基线同样共享、同样可被 Init），属存量行为；转换失败回落为本 PR 新增暴露面，
  二期 reserve 拒 Init 防护落地后此条失效；
- logicId 键（SocketProcess）：越界/转换失败回落 0 号设备子树（与基线 GetInstance 槽位 0
  语义一致，不用 reserve）；
- GlobalNetDevMgr：转换或 Init 失败返回该 phyId 子树的未 Init 实例（失败按设备隔离，
  无共享兜底对象）。
- 析构期回落语义：DeInit 后（退役区析构 / RESET 残留）回落 reserve_ 或 0 号子树时，清理为静默
  no-op——设备侧归还在 RESET 下已被跳过，析构期保证不崩溃、不保证归还（与 R5 已知限制同源，
  二期析构抑制落地后此条失效）。

回落路径必须打 WARNING/ERROR 日志。

#### R10 收编单例前反向排查析构触树依赖

把单例收编入树、或把其调用点改走窄访问器之前，必须**反向排查**：全仓扫描谁的
析构函数会经窄访问器触树（`~Xxx()` 内调 `GetXxxMgr(...)` / `GetBaseCommResPerDev(...)`）。
留在树外的 static 单例与树根的析构次序取决于首触顺序、无契约保障——若其析构经
窄访问器重入 `BaseCommMgr::GetInstance()`，两条路径都是 UB：先于树根构造的
退出时晚于树根析构，命中已析构的互斥量与 map；晚于树根析构的则回落 reserve
兜底空表，清理静默失效。
处置三选一：
1. 收编为树值成员，由声明序（R3）保证它死于所触节点之前（首选）；
2. 按 CcuInstance 模式在 Init 期缓存依赖指针，析构只用缓存、绝不经全局访问器；
3. 维持锚定单例（R8），并在 R1 例外清单登记现状与处置计划。
禁止只改调用点不改析构链的"半收编"（SocketProcess 曾踩中此坑，已按处置 1 收编为
BaseCommResPerDev 值成员）。
锚定单例（晚于树死）的析构链触树点，处置 2 的缓存模式不适用（缓存目标随树先死、
缓存指针同样悬垂）：必须经 `hcomm::IsResourceTreeTeardownFinished()` 判活守卫，
标志置位后跳过归还、仅清本地账（首个实例：CcuComponent::ReleaseAllTpInfos）。
