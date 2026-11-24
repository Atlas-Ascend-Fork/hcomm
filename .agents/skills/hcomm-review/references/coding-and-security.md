# 检视规范：编码与安全

面向 PR 检视与全仓检视的编码规范要点（C/C++），供检视者与 AI Agent 逐条对照。规范全集以根目录 `AGENTS.md` 第 5 节、`.clang-format` 与 [CANN 编码规范](https://gitcode.com/cann/community/tree/master/contributor/coding-standards)为准。本文整合了 HCCL/Hcomm 项目安全编码规范（鉴权与安全配置、资源管理、指针校验、数值安全、命令注入防护、拒绝服务防护、环境变量校验等），按风险等级与检视要点逐条列出。

---

## 检视红线

- **基于代码事实**：所有结论须实读代码验证（读完整函数体而非仅 diff、grep 调用方），严禁凭推测报告
- **只读不改**：检视过程禁止修改被检视代码、禁止写入目标仓任何文件；产出只有检视意见评论
- 检视前先读变更函数的完整实现与所属类头文件，理解上下文再判断（diff 只有片段，易断章取义）
- **只分析对系统产生真实影响的问题** — 排除理论风险、假设性风险、无实际触发路径的风险
- **潜在风险降级**：当前代码在当前运行条件下不会出问题，仅因后续修改/新增代码才可能触发的问题，一律判定为低风险，并标注"潜在风险：仅当后续修改/新增代码时才可能触发"

---

## 命名与风格

| 检查项 | 要求 |
|--------|------|
| 类/函数 | PascalCase；成员变量小驼峰 + 尾下划线（`memberVar_`） |
| 常量/宏 | 常量与宏均 `UPPER_SNAKE_CASE`（全大写 + 下划线） |
| 文件名 | 小写 + 下划线 |
| 格式 | 120 列、4 空格、K&R 大括号、指针右对齐（`.clang-format` 为准） |
| 函数长度 | 超过 50 行建议拆分；参数超过 6 个建议封装结构体 |
| 清晰度 | 公共接口须有注释；避免魔法数字（用命名常量）；无歧义命名 |

---

## 敏感信息拦截

上库代码中注释、源码、打印等均不能涉及敏感信息。个人敏感信息包含：姓名、邮箱、工号、手机号、身份证号等能标明个人身份的信息。同时禁止包含公网 IP 地址、公网 URL 地址、域名、公网邮箱地址等。

---

## 内存与资源安全（红线，命中即 CRITICAL/HIGH）

### 速查表

| 检查项 | 检视要点 |
|--------|---------|
| 未初始化 | 变量/结构体使用前必须初始化（`{}` 值初始化或构造函数） |
| 空指针 | 指针解引用前判空；返回指针的函数须读实现确认 null 路径 |
| 越界 | 数组索引、`memcpy`/`memcpy_s` 长度须有边界校验 |
| 除零 | 除法/取模运算除数须保证非零 |
| 整数溢出 | 有符号运算不溢出、无符号不回绕；窄类型与 `size_t` 混算须显式转换 |
| 资源泄漏 | 所有路径（含错误路径）Init/Cleanup 对称；RAII 或智能指针管理 |
| sizeof 误用 | 禁止对指针变量 sizeof 取数组大小；容器对象 sizeof ≠ 数据大小 |
| 禁用函数 | `memcpy`/`sprintf`/`strcpy` 用 `_s` 安全版本替代并检查返回值 |

---

### 规则：动态内存必须配对释放，禁止内存泄露

**风险等级**：**高风险（High）**

malloc/calloc/realloc/new 分配的内存，必须在所有退出路径上配对 free/delete。HCCL/Hcomm 作为长期运行的通信库，内存泄露会随训练步数累积，最终耗尽内存。设备内存（HrtMalloc/hrtMalloc）同样适用，必须配对 HrtFree/hrtFree。连续多次分配中途失败时，必须回滚前面已分配的内存。

| 检查项 | 检视方法 |
|--------|----------|
| malloc/calloc/new 后所有 return/goto/break 前是否 free/delete | 对每个分配点，列出函数内所有退出路径，确认每条路径在 free 之后 |
| 连续多次分配，中途失败是否回滚前面已分配的 | 搜索连续 malloc/calloc，确认中途失败时 goto cleanup 或逐个回滚 |
| 裸 new（非智能指针）是否有配对 delete | 搜索 `new\s+\w`（含小写基本类型 `new char/int`），确认有 delete 或被智能指针管理 |
| 设备内存 HrtMalloc/hrtMalloc 是否配对 HrtFree | 搜索 HrtMalloc，确认成员变量由析构 HrtFree，局部变量在函数内 HrtFree |
| malloc 返回值可能 NULL 的函数调用是否检查 | 搜索返回指针的函数调用（strdup/RankToString 等），使用前检查 NULL |

**适用范围**：
- 主机内存：malloc/calloc/realloc/strdup + free，new/delete
- 设备内存：HrtMalloc/hrtMalloc + HrtFree/hrtFree，DeviceMem::alloc
- 连续分配场景：函数内多次分配，中途失败需回滚
- 裸指针 new：非 unique_ptr/shared_ptr 管理的 new（含 `new char[]` 等基本类型）

---

### 规则：文件描述符与句柄必须配对关闭，成功路径亦然

**风险等级**：**高风险（High）**

open/fopen/fdopen 等创建的文件描述符/FILE*，必须在所有路径（含成功路径）配对 close/fclose。常见错误：异常路径正确关闭，但成功路径遗漏关闭。推荐使用 goto cleanup 统一释放。

| 检查项 | 检视方法 |
|--------|----------|
| open/fopen 后所有 return 前是否 close/fclose | 对每个 open/fopen，列出函数内所有 return（含成功 return），确认每条都 close/fclose |
| 成功路径是否遗漏关闭（最易漏） | 重点检查函数末尾的正常 return 语句前是否 close/fclose |
| 是否使用 goto cleanup 统一释放 | 推荐所有退出路径 goto 到统一 cleanup 标签 close |
| fd 通过出参返回的，调用方是否 close | 确认出参 fd 的调用方在所有路径 close |

**适用范围**：
- 文件打开：open/fopen/fdopen/creat + close/fclose
- socket：socket/accept + close
- 所有路径：含成功路径、错误路径、异常路径

---

### 规则：异常分支必须释放已分配资源（多退出点防护）

**风险等级**：**高风险（High）**

函数中存在多个退出点（return/CHK_RET/CHK_PTR_NULL/goto）时，每个退出点都必须释放已分配资源。推荐使用 goto cleanup 统一释放（C 代码），或 RAII/智能指针自动管理（C++ 代码）。常见错误：错误处理宏前遗漏 free；函数返回值可能 NULL 未检查，传入下游函数崩溃，导致已分配内存无法到达 free 路径。

| 检查项 | 检视方法 |
|--------|----------|
| 每个退出点（return/CHK_RET/CHK_PTR_NULL/goto）前是否释放已分配资源 | 列出函数所有退出点，逐一确认已分配资源已释放 |
| 函数返回值可能 NULL 的调用是否检查 | 搜索返回指针的函数调用（strdup/RankToString/NetLayerToString 等），使用前检查 NULL |
| 多次分配是否用 goto cleanup 统一释放 | 推荐所有错误路径 goto 到 cleanup 标签，统一 free |
| C++ 是否用 RAII/智能指针管理动态资源 | 推荐 unique_ptr/shared_ptr，异常安全 |

**适用范围**：
- 多退出点函数：含多个 return/CHK_ 宏的函数
- 连续分配：函数内多次 malloc/new，中途失败需释放前面已分配的
- 下游返回值可能 NULL：调用返回指针的函数，使用前必须检查 NULL
- C/C++ 通用：C 用 goto cleanup，C++ 用 RAII/智能指针

---

### 规则：成对资源 API（Open/Init/Create）必须配对 Close/DeInit/Destroy

**风险等级**：**高风险（High）**

HCCL/Hcomm 大量使用成对资源 API，Open/Init/Create 后必须配对 Close/DeInit/Destroy。这类接口管理的资源（句柄、设备内存、通信通道）不走 malloc/free 路径，常规内存泄露扫描无法发现。常见错误：Open/Init/Create 成功后，异常路径未配对 Close/DeInit/Destroy；或析构函数的状态守卫跳过已分配资源的释放。

**项目中的成对资源 API**（非穷举）：

| 获取（Open/Init/Create） | 释放（Close/DeInit/Destroy） | 资源类型 |
|------------------------|---------------------------|----------|
| `hrtIpcOpenMemory` | `hrtIpcDestroyMemoryName` | IPC memory 句柄 |
| `hrtIpcOpenNotify`/`hrtIpcOpenNotifyWithFlag` | `hrtNotifyDestroy` | IPC notify 句柄 |
| `HrtRaInit` | `HrtRaDeInit` | RA 上下文 |
| `HrtRaRdmaInit` | `HrtRaRdmaDeInit` | RDMA handle |
| `HrtRaUbCtxInit` | `HrtRaUbCtxDestroy` | UB context handle |
| `HrtRaSocketInit` | `HrtRaSocketDeInit` | socket handle |
| `HrtRaQpCreate`/`HrtRaNdaQpCreate` | `HrtRaQpDestroy` | QP handle |
| `HrtRaNdaCqCreate` | `HrtRaNdaCqDestroy` | CQ handle |
| `HrtRaTlvInit` | `HrtRaTlvDeInit` | TLV handle |
| `HrtNotifyCreate`/`HrtCntNotifyCreate` | `HrtNotifyDestroy`/`HrtCntNotifyDestroy` | notify handle |
| `RaLoopbackQpCreate` | `RaQpDestroy` | loopback QP |

| 检查项 | 检视方法 |
|--------|----------|
| Open/Init/Create 后所有退出路径是否配对 Close/DeInit/Destroy | 对每个成对 API 调用，列出函数内所有退出路径，确认每条路径在释放之后 |
| 析构函数是否能兜底（状态守卫是否跳过清理） | 检查析构函数/DestroyAll 的状态守卫，确认失败状态下不会跳过已分配资源的释放 |
| 连续 Create（Create A + Create B）中途失败是否释放 A | 搜索连续 Create（CqCreate+QpCreate 等），确认中途失败时 Destroy 已创建的 |
| handle 存入 map/成员变量的，清理函数是否遍历释放 | 确认 DestroyAll/析构遍历 map 对每个 handle 释放 |
| 成对 API 清单识别 | 搜索 `[hH]rt\w*(Open\|Close\|Init\|DeInit\|Create\|Destroy)`（注意：实际执行时需去掉正则中的 `\` 转义，使用 `[hH]rt\w*(Open|Close|Init|DeInit|Create|Destroy)`）、`aclrt\w*(Create\|Destroy)`（同上，实际用 `aclrt\w*(Create|Destroy)`） |

**适用范围**：
- hrtIpc 系列、HrtRa 系列、aclrt 系列
- 项目自定义成对 API：所有 Open/Close、Init/DeInit、Create/Destroy、Register/Unregister、Acquire/Release 模式

---

## 鉴权与安全配置

HCCL/Hcomm 项目中 UB（Unified Bus）访问协议的鉴权安全配置，涉及 token 校验机制。以下规则按各自标注的风险等级执行检视，

### 规则：禁止在 UB 通信中将 token_policy 设置为 URMA_TOKEN_NONE

**风险等级**：**致命（Critical）**

在 UB 访问协议中，token 是用于鉴权的关键字段，用于验证通信请求的合法性，防止未授权的设备或进程访问通信资源。token 校验机制确保身份认证、访问控制、隔离保护。禁止将 `token_policy`/`tokenPolicy` 设置为 `URMA_TOKEN_NONE`/`TOKEN_POLICY_NONE`/`MEM_SEG_TOKEN_NONE`（关闭 token 校验），否则将导致未授权访问、数据泄露、数据篡改/注入、资源劫持、横向渗透等致命安全风险。

**字段命名分裂**（全仓搜索验证）：同一语义位域 `token_policy : 3` 在代码中存在两套字段名拼写：

| 拼写形态 | 定义位置 | 使用文件数 | 赋值行数 | 说明 |
|:---|:---|:---|:---|:---|
| `token_policy`（snake_case） | `urma_types.h:579/660/793/836/861` | 2 | 4 | URMA 上游字段名 |
| `tokenPolicy`（camelCase） | `hccp_ctx.h:147`（`RegSegFlag`）、`hccp_ctx.h:434`（`ImportJettyFlag`） | 3 | 7 | HCCP 本地镜像字段名 |

**常量族三重定义**（src/ 内搜索验证）：同一语义取值存在三套等价常量，数值一一对应：

| 常量族 | 定义位置 | 定义类型 | 使用文件数 | 使用行数 | 服务字段 |
|:---|:---|:---|:---|:---|:---|
| `URMA_TOKEN_*` | `urma_opcode.h:16-20` | `#define` 宏 | 2 | 4 | URMA 上游 `token_policy` |
| `TOKEN_POLICY_*` | `hccp_ctx.h:425-429` | 匿名 enum | 2 | 6 | HCCP 本地 `tokenPolicy` |
| `MEM_SEG_TOKEN_*` | `hccp_ctx.h:131-135` | 具名 enum `MemSegTokenPolicy` | 0 | 0 | src/ 内零使用，仅 test/st 有 1 处使用（`test/st/hccp/test_case/cases_ub_ctrl.cc:505`） |

> **注意**：`RegSegFlag::tokenPolicy` 注释（`hccp_ctx.h:147`）引用的 `enum mem_seg_token_policy` 不存在，真实 enum 名为 `MemSegTokenPolicy`（`hccp_ctx.h:130`），该 enum 在 src/ 内零使用（仅定义处），test/st 有 1 处使用（`test/st/hccp/test_case/cases_ub_ctrl.cc:505`，`MEM_SEG_TOKEN_PLAIN_TEXT`）。实际赋值用的常量来自匿名 enum `TOKEN_POLICY_*`（与 `URMA_TOKEN_*` 数值一一对应）。

**token_policy 取值范围**（常量定义于 `urma_opcode.h:16-20`，`urma_jfr_flag` 结构体定义于 `urma_types.h:577-595`，`token_policy` 位域在 `:579` 行）：

| `URMA_TOKEN_*` | `TOKEN_POLICY_*` | `MEM_SEG_TOKEN_*` | 值 | 含义 | 安全等级 |
|------|------|------|---|------|----------|
| `URMA_TOKEN_NONE` | `TOKEN_POLICY_NONE` | `MEM_SEG_TOKEN_NONE` | 0 | 关闭 token 校验 | **致命风险，禁止使用** |
| `URMA_TOKEN_PLAIN_TEXT` | `TOKEN_POLICY_PLAIN_TEXT` | `MEM_SEG_TOKEN_PLAIN_TEXT` | 1 | 明文 token 校验 | 合规（最低安全要求） |
| `URMA_TOKEN_SIGNED` | `TOKEN_POLICY_SIGNED` | `MEM_SEG_TOKEN_SIGNED` | 2 | 签名 token 校验 | 合规（推荐生产环境） |
| `URMA_TOKEN_ALL_ENCRYPTED` | `TOKEN_POLICY_ALL_ENCRYPTED` | `MEM_SEG_TOKEN_ALL_ENCRYPTED` | 3 | 全加密 token 校验 | 合规（最高安全等级） |
| `URMA_TOKEN_RESERVED` | `TOKEN_POLICY_RESERVED` | `MEM_SEG_TOKEN_RESERVED` | 4 | 保留值 | 禁止使用 |

> **注意**：`URMA_TOKEN_STRICT` 和 `URMA_TOKEN_BASIC` 在代码中不存在。常量定义在 `src/base_comm/resources/hccp/external_depends/ubengine/urma_opcode.h:16-20`，结构体定义在 `src/base_comm/resources/hccp/external_depends/ubengine/urma_types.h:577-595`（`token_policy` 位域在 `:579` 行）。`token_policy` 还出现在 `urma_types.h` 的 660、793、836、861 行（多个结构体共用同一位域定义）。HCCP 本地镜像字段 `tokenPolicy` 定义在 `src/base_comm/resources/hccp/inc/network/hccp_ctx.h:147`（`RegSegFlag`）和 `:434`（`ImportJettyFlag`），配套常量 `TOKEN_POLICY_*` 定义在 `hccp_ctx.h:425-429`。

| 检查项 | 检视方法 |
|--------|----------|
| token_policy / tokenPolicy 是否为 NONE | 搜索 `token_policy`、`tokenPolicy`、`URMA_TOKEN_NONE`、`TOKEN_POLICY_NONE`、`MEM_SEG_TOKEN_NONE`，确保所有设置均非 NONE |
| token_policy / tokenPolicy 的合理取值 | 确认使用 `URMA_TOKEN_PLAIN_TEXT`(1) / `TOKEN_POLICY_PLAIN_TEXT`(1) / `URMA_TOKEN_SIGNED`(2) / `URMA_TOKEN_ALL_ENCRYPTED`(3)，最低要求为 PLAIN_TEXT |
| 通信初始化函数中的鉴权配置 | 搜索所有 `.bs.` 结构体初始化，检查 `token_policy` / `tokenPolicy` 字段 |
| URMA_TOKEN_NONE / TOKEN_POLICY_NONE 是否出现在生产代码 | 搜索所有 `URMA_TOKEN_NONE`、`TOKEN_POLICY_NONE`、`MEM_SEG_TOKEN_NONE` 出现位置，逐一确认是否为测试代码 |
| SegCb / Channel 相关初始化 | 搜索 `RsUbCtxJettyCreate`、`RsUbCtxInit`、`HccpUbCreateJetty` 等初始化函数，检查 token 校验配置 |
| tokenValue 是否泄露到日志 | 搜索日志宏中是否包含 `tokenValue`（参见「禁止将 tokenValue 输出到任何等级日志中」规则） |

**适用范围**：
- Segment 初始化：如 `RsUbCtxJettyCreate`、`RsUbCtxInit`（`rs_ub.c`）等 UB 通信初始化函数
- Transport 初始化：所有创建 UB 通信连接的初始化代码
- 测试代码：即使是测试代码，使用 `URMA_TOKEN_NONE` 也需明确标注测试用途和安全免责声明

---

### 规则：禁止将 tokenValue 输出到任何等级日志中

**风险等级**：**致命（Critical）**

在 UB 访问协议中，tokenValue 是内存访问鉴权信息，由随机数生成，每次生成的 tokenValue 不同。将 tokenValue 输出到日志会导致鉴权凭证泄露、持久化泄露、攻击链放大、跨节点扩散。

**核心安全规则**：
- tokenValue 禁止输出到任何等级日志中（包括 DEBUG、INFO、WARN、ERROR 等所有级别）
- tokenValue 在代码中主要以 `uint32_t`（含 `u32` 别名）类型存在，在 CCU 相关代码中存在 `uint64_t` 变体；在 urma C API 中以 `urma_token_t`（内部 `uint32_t token`）类型存在。无论以整数形式还是字符串形式输出，均属致命安全违规
- tokenValue 的实际变量形态（均经全仓搜索验证，以下形态真实存在于 `src/` 中）：
  - `tokenValue`（驼峰，主流形态，遍布 `src/` 多数目录）
  - `token_value`（下划线，urma C API 字段名，定义于 `urma_types.h:635/806/920/961/1083` 等，在 `rs_ub.c`/`rs_ping_urma.c` 中作为 jfrCfg/segCfg 字段赋值）
  - `attr.ub.tokenValue`（urma 适配器中的结构体成员访问路径，分布于 `hcomm_adapter_hccp.cc`、`orion_adapter_hccp.cc`）
  - `dbTokenValue`（CCU jetty 配置字段，`ccu_jetty_ctx_mgr.cc:65-67` 拆分为 `tokenValueLow`/`Middle`/`High`；定义于 `ccu_device_manager.h:53`、`ccu_dev_mgr_imp.h:51`）
  - `rmtTokenValue`（远端 token 凭证值，`aicpu_task_cache_entry.cc:751`、`ub_conn_lite.cc:97`、`udma_data_struct.h:88/146`、`mc2_type.h:239` 等）
  - `notifyTokenValue`（notify 通道 token 凭证值，`ub_conn_lite.cc:496`、`udma_data_struct.h:110`）
  - `memTokenValue`（CCU 内存 token 凭证值，`ccu_comp.cc:670`、`ccu_connection.cpp:410`、`ccu_device_manager.h:64`）
  - `ccuBufTokenValue`（CCU buffer token 凭证值局部变量，`ccu_comp.cc:473/659`、`ccu_connection.cpp:91/211`（legacy 路径 `src/legacy/ascend950/unified_platform/resource/ccu_transport/`，非 `ccu_component.cpp`/`ccu_conn.cc`）、`ccu_connection.h:87`）
  - `GetTokenValue()`（RMA buffer 取值方法，`local_ub_rma_buffer.cc:173`、`remote_rma_buffer.h:128`、`rmt_rma_buf_slice_lite.h:29` 等多处）
  - `protectionInfo.memInfo.ub.tokenValue`（CCU/URMA 通道结构体成员访问路径，`aicpu_ts_channel_entity_resources.cc:96`、`ccu_instance_snapshot.cc:116`、`ccu_urma_channel.cc:527/580`、`aiv_urma_transport.cc:327/343` 等）
- tokenValue 可能被隐式转换为其他变量（赋值、拷贝、格式化等），所有由 tokenValue 赋值的变量同样禁止输出到日志

> **注意：以下形态在 `src/` 中不存在，属于虚构项，检视时勿引用**：`token.val`、`tokenvalue`（全小写）、`tokenval`（无分隔）。  
> **注意：`tokenValue` 类型不限于 `uint32_t`**：CCU 上下文存在 `uint64_t tokenValue` 变体（如 `ccu_launch.cc:471`、`ccu_kernel_mgr.cc:912`）；urma C API 中为 `urma_token_t`（其内部为 `uint32_t token`，定义于 `urma_types.h:316-318`）。
> **注意：RDMA rkey/lkey 不属于本规则约束范围**（详见「RDMA rkey/lkey 与 UB tokenValue 的区分」规则）。

**tokenValue 变量传播规则**：

| 传播方式 | 安全要求 |
|----------|----------|
| 直接引用（`attr.ub.tokenValue`） | 禁止输出到日志 |
| 赋值传递（`uint32_t tv = tokenValue;`） | `tv` 同样禁止输出到日志 |
| 结构体成员赋值（`jfrCfg.token_value.token = ...`） | 禁止输出到日志 |
| 命名变体（`tokenValue`、`token_value`） | 禁止输出到日志 |
| 内存拷贝（`memcpy_s` 含 tokenValue 的缓冲） | 包含 tokenValue 的部分禁止输出到日志 |
| 格式化传递（`sprintf_s` 含 tokenValue） | 包含 tokenValue 的部分禁止输出到日志 |
| 结构体赋值 | 下游变量禁止输出到日志 |

**配置传递说明**：tokenValue 允许作为配置项传递（结构体成员赋值、函数参数传递、进程间数据交换等），这是正常业务数据流，不属于安全违规。安全约束仅针对日志输出。但传递路径上的所有下游代码仍需确保不会将 tokenValue 输出到日志。

| 检查项 | 检视方法 |
|--------|----------|
| tokenValue 是否出现在任何日志输出 | 搜索日志宏中是否包含 `tokenValue`、`token_value` 或 `TokenValue`（大小写不敏感，覆盖 `dbTokenValue`/`rmtTokenValue`/`notifyTokenValue`/`memTokenValue`/`ccuBufTokenValue`/`GetTokenValue` 等首字母大写变体）。**仅排除有效性标志噪声项**：`grep -inP "token_value|tokenvalue(?!isvalid)" *.c *.cc *.cpp *.h`。**禁止用行级 `grep -v "tokenValueIsValid"`**——`grep -v` 按整行丢弃，当同一行同时出现 `tokenValueIsValid`（有效性标志）与 `tokenValue`（凭证泄露）时会整行丢弃、掩盖泄露，与"保留 `GetTokenValue`/`ccuBufTokenValue` 等变体命中供上下文检视、仅排除 `tokenValueIsValid` 噪声"的混合输出要求矛盾；项级负向断言 `(?!isvalid)` 只丢弃噪声匹配项、保留同行其它命中，可正确混合输出。`tokenValueIsValid` 是有效性标志（允许输出，参见「tokenValueIsValid 允许输出到日志，但下发/使用前必须赋值为 TOKEN_VALUE_IS_VALIDE」规则），非凭证值本身，可排除。**`tokenValueLow`/`Middle`/`High` 及 `dstTokenValueLow`/`Middle`/`High` 是 `dbTokenValue`/`memTokenValue` 的位域分片，属于凭证数据的组成部分，禁止排除** |
| tokenValue 的赋值变量是否出现在日志 | 搜索所有由 tokenValue 赋值的变量名，追踪这些变量是否出现在日志输出中 |
| printf/sprintf 等格式化函数中的 tokenValue | 搜索 `printf`、`sprintf`、`fprintf`、`snprintf` 等函数参数中是否包含 tokenValue 或其赋值变量 |
| UB 通信初始化/创建函数中的日志 | 搜索 `HccpUbCreateJetty`、`RsUbCtxJettyCreate`、`RsUbCtxInit` 等函数中的日志 |
| 日志级别是否包含 tokenValue | 检查所有级别的日志（包括 DEBUG），确认均不含 tokenValue |
| attr.ub 结构体成员的日志输出 | 搜索日志中输出 `attr.ub` 成员的地方，确认不含 tokenValue |
| 排除 tokenValueIsValid 噪声字段 | 检索 `tokenValue` 时必须用 `grep -inP`（大小写不敏感 + PCRE）并以项级负向断言 `(?!isvalid)` 排除 `tokenValueIsValid`（有效性标志，非凭证值，不属于本规则约束范围），**禁止用行级 `grep -v` 排除**（行级 `grep -v` 会整行丢弃，掩盖同行内真实的 `tokenValue` 凭证泄露，详见上行 tokenValue 检索项）。**不得排除 `tokenValueLow`/`Middle`/`High` 及 `dstTokenValueLow`/`Middle`/`High`**——它们是 `dbTokenValue`/`memTokenValue` 的位域分片，属于凭证数据的组成部分，输出到日志同样构成泄露 |

**适用范围**：
- 通信初始化：如 `HccpUbCreateJetty`、`RsUbCtxJettyCreate`、`RsUrmaCreateJetty` 等函数中的日志输出
- Channel/Segment 创建：如 `HcommChannelCreate`、`HcclChannelAcquire` 等函数中的日志
- 错误处理日志：即使创建失败，错误日志中也不得输出 tokenValue
- 调试日志：DEBUG 级别日志同样禁止输出 tokenValue
- 诊断/维测代码：任何诊断维测代码中均禁止输出 tokenValue

---

### 规则：RDMA rkey/lkey 与 UB tokenValue 的区分（避免误报）

**风险等级**：**澄清说明**（非风险规则，用于避免检视误报）

在 HCCL/Hcomm 代码检视中，RDMA 通信相关字段（如 `rkey`、`lkey`）与 UB 通信相关字段（如 `tokenValue`）可能出现在相同的代码上下文中，容易混淆。检视者可能误将 rkey/lkey 视为敏感信息，从而产生误报。

| 对比项 | RDMA rkey/lkey | UB tokenValue |
|--------|---------------|---------------|
| 定义 | 内存注册后驱动分配的地址访问索引 | UB 通信鉴权凭证 |
| 性质 | 地址描述符，标识已注册内存区域 | 鉴权凭证，验证通信请求合法性 |
| 生成方式 | 由 RDMA 驱动在内存注册（ibv_reg_mr）时分配 | 随机数生成，每次通信创建时不同 |
| 是否敏感 | 否，通常可输出到日志用于调试 | 是，禁止输出到任何等级日志 |
| 泄露风险 | 低，仅在已建立连接的通信双方间有效，脱离连接上下文无法利用 | 高，泄露后可被用于伪造鉴权、未授权访问 |
| 安全要求 | 可正常打印到日志（建议仅 DEBUG 级别） | 绝对禁止打印到任何级别日志（参见「禁止将 tokenValue 输出到任何等级日志中」规则） |

| 检查项 | 检视方法 |
|--------|----------|
| rkey/lkey 输出到日志 | 允许，不属于安全违规，无需报出 |
| tokenValue 输出到日志 | 禁止，属于致命安全违规，参见「禁止将 tokenValue 输出到任何等级日志中」规则 |
| 混淆 rkey/lkey 与 tokenValue | 仔细区分变量名：`rkey`、`lkey` 为 RDMA 字段（可打印）；`tokenValue` 为 UB 鉴权字段（禁止打印） |
| 其他含 key 字段 | 根据上下文判断：若为 RDMA 内存注册相关（mr->rkey、mr->lkey），允许打印；若为 UB/URMA 鉴权相关（tokenValue），禁止打印 |

---

### 规则：tokenValueIsValid 允许输出到日志，但下发/使用前必须赋值为 TOKEN_VALUE_IS_VALIDE

**风险等级**：**高风险（High）**（如果 tokenValueIsValid 在 jetty 上下文下发/使用时仍为 0，意味着 token 校验未生效，等同于「禁止在 UB 通信中将 token_policy 设置为 URMA_TOKEN_NONE」规则中关闭 token 校验的效果）

`tokenValueIsValid` 是 UB 通信中用于标记 tokenValue 鉴权凭证是否有效的标志字段。取值含义（仓内仅定义 `TOKEN_VALUE_IS_VALIDE = 1`，`ccu_jetty_ctx_mgr.h:23`；值为 0 时无对应命名常量）：

| 值 | 含义 | 安全影响 |
|---|------|----------|
| 0（无命名常量；仓内仅定义 `TOKEN_VALUE_IS_VALIDE = 1`） | tokenValue 无效，token 校验未生效 | 高风险：等同于关闭 token 校验，攻击者可绕过鉴权 |
| 1（`TOKEN_VALUE_IS_VALIDE`） | tokenValue 有效，token 校验已启用 | 安全合规：鉴权机制正常运作 |

**安全规则**：
1. tokenValueIsValid 允许输出到日志：与 tokenValue（鉴权凭证值）不同，tokenValueIsValid 是鉴权有效性标志，属于状态信息而非凭证本身
2. tokenValueIsValid 在 jetty 上下文配置完成（下发/使用）前必须赋值为 `TOKEN_VALUE_IS_VALIDE`：如果在使用时仍为 0，表示 token 校验未生效，其安全影响等同于「禁止在 UB 通信中将 token_policy 设置为 URMA_TOKEN_NONE」规则中将 `token_policy` 设置为 `URMA_TOKEN_NONE`
3. 结构体默认初始化 `{0}` 的中间态不属违规：仓内 `tokenValueIsValid{0}`（`ccu_jetty_ctx_mgr.h:83`）是结构体构造的默认初值，创建流程中后续会赋值为 `TOKEN_VALUE_IS_VALIDE`（`ccu_jetty_ctx_mgr.cc:60`），此默认初始化中间态不属安全违规
4. tokenValueIsValid 在下发/使用时为 0 的场景需要明确标注原因：如在测试代码中，必须明确标注测试用途和安全免责声明

**与前序规则的关系**：

| 规则 | 约束对象 | 约束内容 |
|------|----------|----------|
| 禁止在 UB 通信中将 token_policy 设置为 URMA_TOKEN_NONE | `token_policy` | 禁止设置为 `URMA_TOKEN_NONE`（关闭 token 校验） |
| 禁止将 tokenValue 输出到任何等级日志中 | `tokenValue` | 禁止输出到任何等级日志（鉴权凭证泄露） |
| tokenValueIsValid 允许输出到日志，但下发/使用前必须赋值为 TOKEN_VALUE_IS_VALIDE | `tokenValueIsValid` | 允许输出到日志，但下发/使用前必须赋值为 `TOKEN_VALUE_IS_VALIDE`（token 校验生效） |

> **注意**：`tokenValueIsValid` 和 `tokenValue` 是两个不同字段。`tokenValueIsValid` 是鉴权有效性标志（允许输出），`tokenValue` 是鉴权凭证值（禁止输出）。检视时需区分两者，避免将 `tokenValueIsValid` 的日志输出误报为 `tokenValue` 泄露。

| 检查项 | 检视方法 |
|--------|----------|
| tokenValueIsValid 在下发/使用时是否仍为 0 | 搜索 `tokenValueIsValid` 的赋值，确认在 jetty 上下文下发/使用前已赋值为 `TOKEN_VALUE_IS_VALIDE`（`ccu_jetty_ctx_mgr.cc:60`）；排除结构体默认初始化 `{0}` 的中间态（`ccu_jetty_ctx_mgr.h:83`），该中间态不属违规 |
| tokenValueIsValid 输出到日志 | 允许，不属于安全违规（与 tokenValue 禁止输出不同） |
| tokenValueIsValid 与 tokenValue 混合输出 | 搜索日志中同时包含 `tokenValueIsValid` 和 `tokenValue` 的场景，确认 tokenValue 未输出 |

**适用范围**：
- Jetty 上下文配置：如 `ConfigJettyCtxData`、`DumpJettyCtxData` 等函数
- Channel 初始化：如 UB 通信创建过程中的鉴权有效性标记
- 日志输出：tokenValueIsValid 输出到日志是允许的，但需确认 jetty 上下文下发/使用前已赋值为 `TOKEN_VALUE_IS_VALIDE`

---

## 指针与参数校验

> **强制检视顺序**：空指针检视必须按以下优先级顺序执行，不得跳级：
> ```
> 第1级：参数有效性检查 — 直接解引用/确定解引用（致命） ← 最高优先级，必须首先完成
> 第2级：参数有效性检查 — 仅传递给函数（低风险，默认；有确定触发路径时升级为高风险）
> 第3级：工厂方法/查找函数返回值检查（高风险）
> 第4级：获取结果再次确认（高风险）
> 第5级：后续使用检查（低风险）
> ```

### 规则：函数的指针参数必须在首次使用前检查 nullptr（最高优先级）

**风险等级**：取决于首次使用方式（见风险等级判定规则）

**检视优先级**：**最高**（必须在「工厂方法返回值与 dynamic_cast 结果必须做空指针检查后再使用」、「公开 API 结构体指针成员使用前必须零初始化」、「内存分配函数返回值必须做空指针检查，free/delete 后指针必须立即赋予新值」规则之前检查）

函数的指针参数可能为 nullptr，必须在首次使用前检查，不得"先使用后检查"。这是空指针检视中最高优先级的问题，必须优先于工厂方法返回值检查等其他检查。如果指针参数为 nullptr，在首次使用（日志打印、函数调用、解引用）时会导致进程崩溃（SIGSEGV）、未定义行为、防御失效（后续的参数检查无法补救已经发生的崩溃）。

#### 风险等级判定规则（强制执行）

指针参数未检查 nullptr 即首次使用的风险等级，取决于首次使用的方式：

**直接解引用 / 确定会解引用的调用：致命（Critical）或高风险（High）**
- 直接解引用：`param->field`、`*param`、`param[i]`
- 日志打印中作为 %s 参数：`HCCL_INFO("... %s", param)` — printf("%s", nullptr) 是未定义行为
- 字符串操作函数：`strlen(param)`、`strcpy(dst, param)`、`strcmp(param, ...)` — C 标准库函数确定会解引用
- 当前代码中可确定被调用函数会解引用该指针

**仅传递给函数 / 组件，未解引用：低风险（Low）（默认）**
- 仅传递给函数作为参数：`SomeFunction(param, ...)` — 当前代码未解引用，仅传递指针值
- 仅赋值给变量/结构体成员：`obj.field = param`
- 仅作为函数返回值传递：`return param`

**低风险升级为高风险的条件**：当且仅当满足以下全部条件时，升级为高风险：
1. 当前代码存在确定的执行路径，被调用函数/组件会解引用该空指针
2. 该执行路径可被实际触发（非假设性/理论性路径）
3. 被调用函数无法优雅处理 nullptr（即被调用函数内部不做 nullptr 检查直接解引用）

**外部组件接口降级规则（强制执行）**：当被调用函数/组件的实现不在当前源码树中时（即外部组件接口），上述升级条件1和3无法确认，默认维持低风险，不得升级为高风险。判定逻辑：
- 若被调用函数实现不在当前源码树中（外部组件接口）：风险等级 = 低风险（Low），不得升级。指针校验由被调用方组件负责。
- 若被调用函数实现可在源码树中找到：按升级条件1-3判定。

**如何判断"实现不在源码树中"**：
- 使用 `grep`/`git grep` 在 src/ 目录下搜索被调用函数名的定义（非调用处），若无结果则为实现不在源码树中
- 若仅在 test/ 目录下找到测试桩（stub），测试桩不等同于真实实现，仍视为"实现不在源码树中"
- 若在 src/ 中找到实现，则按正常升级条件1-3判定

**不适用场景**（仍按正常升级条件判定）：
- 被调用函数实现在源码树中可找到 → 需查看实现是否解引用、是否处理 nullptr，按条件1-3判定
- 被调用函数是 C 标准库函数（如 `strlen`/`strcpy`/`printf`）→ 标准库行为已知（确定解引用），直接判定为致命

**判定示例**：

| 首次使用方式 | 是否解引用 | 默认风险等级 | 说明 |
|-------------|-----------|-------------|------|
| `param->field` | 直接解引用 | 致命 | 确定解引用，崩溃 |
| `HCCL_INFO("... %s", param)` | 确定解引用 | 致命 | printf("%s", nullptr) 是 UB |
| `strlen(param)` | 确定解引用 | 致命 | C 标准库确定解引用 |
| `SomeFunction(param)` | 仅传递 | 低风险 | 当前代码未解引用，仅传递指针值 |
| `obj.field = param` | 仅赋值 | 低风险 | 仅赋值操作，未解引用 |
| `SomeFunction(param)` 且被调用方实现**在源码树中**、确定解引用 | 传递后解引用 | 高风险（需确认触发路径） | 追踪到被调用方确定会解引用且有触发路径 |
| `SomeFunction(param)` 且被调用方实现**不在源码树中**（外部组件接口） | 仅传递，无法确认 | 低风险（不得升级） | 实现不可见，无法确认会解引用；指针校验由被调用方组件负责 |

#### 检视要点（强制执行）

**检视优先级顺序**（必须按此顺序执行，不得跳级）：

```
第1级：参数有效性检查 — 直接解引用/确定解引用（致命） ← 最高优先级，必须首先完成
  ├─ 所有指针参数是否在首次使用前检查 nullptr
  ├─ 是否存在"先使用后检查"违规模式
  └─ 日志打印、解引用、字符串操作中的指针使用（确定解引用的场景）

第2级：参数有效性检查 — 仅传递给函数（低风险，默认） ← 需追踪被调用方判断是否升级
  ├─ 传递给函数/组件但未解引用
  ├─ 追踪被调用方是否解引用该指针
  └─ 仅当有确定触发路径时升级为高风险

第3级：工厂方法/查找函数返回值检查（高风险）
  ├─ new/malloc/calloc 返回值检查
  ├─ 工厂方法返回值检查（GetAlgOp等）
  ├─ 查找函数返回值检查（HcomGetCommHandleByGroup等）
  └─ dynamic_cast 结果检查

第4级：获取结果再次确认（高风险）
  ├─ 工厂方法返回的指针是否为 nullptr
  └─ 查找函数返回的结果是否有效

第5级：后续使用（低风险）
  ├─ 调用其他函数传参
  └─ 解引用操作
```

**风险等级判定流程**（强制执行）：
1. 发现指针参数未检查 nullptr 即首次使用 → 判断首次使用方式
2. 如果是直接解引用 / 确定会解引用的调用（`->field`、`%s`打印、`strlen`等） → 致命/高风险
3. 如果是仅传递给函数/组件（`SomeFunction(param)`、赋值等） → 默认低风险
4. 对低风险项，追踪被调用方实现：
   - 搜索 `git grep -n "{被调用函数名}" {BRANCH} -- "*.c" "*.cc" "*.cpp" "*.h"`
   - 查看被调用方是否解引用该指针
   - 如果被调用方确定解引用且有确定触发路径 → 升级为高风险
   - 如果被调用方内部有 nullptr 检查或无法确定触发路径 → 保持低风险

| 检查项 | 检视方法 | 默认风险等级 |
|--------|----------|-------------|
| 指针参数直接解引用 / printf %s / strlen 等 | 对每个指针参数执行10步污点分析，重点识别首次使用方式 | 致命（最高） |
| 是否存在"先使用后检查"违规模式 | 使用10步污点分析，重点追踪首次使用位置 | 取决于使用方式 |
| 日志打印中作为 %s 参数 | 使用10步污点分析，重点识别风险代码行和被调用方 | 致命 |
| 仅传递给函数/组件，未解引用 | 追踪被调用方实现，确认是否解引用 | 低风险 |
| 传递后追踪到被调用方解引用 | 追踪被调用方实现 + 分析触发路径 | 高风险（有确定路径时） |
| 工厂方法返回值是否检查 | 使用10步污点分析，重点追踪变量传递过程 | 高风险 |
| dynamic_cast 结果是否检查 | 使用10步污点分析，重点列出变量来源和取值 | 高风险 |

#### 污点分析方法（强制执行）

**强制要求**：对所有指针参数问题，必须使用污点分析方法，严格按照以下10步流程执行深度分析。每一步都必须产出明确结果，不得跳过或省略。

1. **识别风险代码行**：明确标注存在风险的代码行（文件路径+行号）、风险类型、首次使用方式（直接解引用/printf %s/strlen 等 → 致命；仅传递给函数/组件 → 低风险默认）
2. **向上追溯所有调用方（直至顶层入口）**：从问题函数出发，逐层向上查找所有调用方，记录函数名、文件路径、行号、参数传递方式
3. **向下追踪被调用方完整链路（直至最底层）**：从问题代码行出发，逐层向下追踪其调用的子函数，记录完整调用链路
4. **列出与风险相关的所有变量/元素**：包括局部变量、全局变量、参数、结构体字段，标注名称、类型、定义位置、初始值
5. **列出与风险相关的控制逻辑和分支取值范围**：分析所有相关的 if/switch/循环等控制逻辑，列出每个分支的取值范围和进入条件
6. **列出与风险相关的计算逻辑和取值影响**：分析算术运算、位运算、指针运算等，说明每个计算结果对风险的影响
7. **列出所有变量在调用方和被调用方的传递过程**：追踪变量从定义→赋值→传递→使用的完整传递链，标注传递方式（值/指针/引用/全局共享）
8. **列出变量的来源和取值**：确定值来源（用户输入/网络/配置/硬编码/计算结果/全局状态），列出所有可能取值，标注哪些取值触发风险
9. **列出变量对传递过程的影响**：分析变量值的变化是否传播到下游并放大风险，识别意外修改
10. **综合分析 — 确定真实问题和触发条件**：基于步骤1~9推导真实问题，列出触发所需的参数条件，只报告具有真实触发路径的问题，给出调用链+数据流+影响范围+修复方案

**辅助命令**:
```bash
# 步骤1: 定位风险代码行
grep -n "{参数名}" {文件} | head -5
# 步骤2: 向上追溯调用方
git grep -n "{函数名}" {BRANCH} -- "*.c" "*.cc" "*.cpp" "*.h"
# 步骤3: 向下追踪被调用方
git grep -n "{被调用函数名}" {BRANCH} -- "*.c" "*.cc" "*.cpp" "*.h"
```

#### 常见违规模式识别（强制搜索）

检视时必须搜索以下违规模式：
- **模式1：日志打印先使用**（最常见，最容易被忽略，致命）— `HCCL_INFO("... %s", param)` 后 `CHK_PTR_NULL(param)`
- **模式2：函数调用先使用（仅传递，低风险默认）**— `SomeFunction(param)` 后 `CHK_PTR_NULL(param)`；追踪被调用方实现，如果确定解引用且有触发路径 → 升级为高风险
- **模式3：解引用先使用**（致命）— `if (param->field > 0)` 后 `CHK_PTR_NULL(param)`
- **模式4：字符串操作先使用**（致命）— `strlen(param)` 后 `CHK_PTR_NULL(param)`

#### 适用范围

- **公开API接口**：如 `HcclCommInitClusterInfo`、`HcclChannelAcquire`、`HcommChannelCreate`、`HcomAllReduce`、`HcomSend`、`HcomReceive` 等
- **内部函数**：所有接收指针参数的内部函数
- **回调函数**：注册的回调函数如果接收指针参数，也必须在首次使用前检查
- **构造函数**：构造函数接收的指针参数，在首次使用前必须检查

#### 误报防范规则（强制执行 — 报告任何指针问题前必须通过以下全部检查）

> **背景**：检视报告中大量"致命问题"为误报。根因包括：虚构代码事实（函数签名/行号/函数边界错误）、漏看已有的跨函数校验（如 `HcomCheckTag`）、对内部函数的指针参数过度要求 nullptr 检查、未追踪指针来源（数组成员/c_str()/make_unique 永远不为 nullptr）。以下 5 条子规则在报告任何指针 nullptr 问题前**必须全部通过**，任一不通过则不得报告该问题。

**子规则A：代码事实验证（防止虚构）**— 在报告任何指针问题前，必须实际读取 cited 行号处的源码，验证：函数签名一致、行号准确（cited "首次使用行"必须是真正的指针使用行，非函数定义行/注释/空行）、函数边界一致（cited "nullptr 检查行"与"首次使用行"在同一函数内）、检查类型正确（cited "nullptr 检查"必须是真正的 nullptr 检查，非大小/范围检查）。

**子规则B：跨函数校验追踪（防止漏看已有检查）**— 在报告"指针参数未检查 nullptr"前，必须搜索首次使用点之前是否存在间接校验调用（`CHK_RET(HcomCheckTag(...))`、`CHK_RET(HcomCheckOpParam(...))`、`CHK_PTR_NULL(...)`、`CHK_PRT_RET(... == nullptr, ...)`、`if (ptr == nullptr)` 等）。若首次使用前存在校验调用且内部执行了 nullptr 检查 → 不得报告。

**HCOMM 常见校验 wrapper 函数清单**（必须了解，避免漏看）：

> 以下函数均在 HCOMM 仓 `src/` 中有真实定义，检视前必须了解以免漏看已有校验导致误报。

**A. legacy ascend910 层综合校验**（定义于 `src/legacy/ascend910/framework/common/src/param_check.cc`）：

- `HcomCheckTag(const char* tag)` — 内部执行 `CHK_PTR_NULL(tag)` + 长度/字符校验
- `HcomCheckCount(u64 count)` — count 范围校验
- `HcomCheckDataType(HcclDataType dataType)` — dataType 合法性校验
- `HcomCheckReductionOp(const std::string& callerOpName, const HcclReduceOp op)` — reduceOp 校验
- `HcomCheckReduceDataType(const HcclDataType dataType, const HcclReduceOp op, DevType deviceType)` — dataType+reduceOp+deviceType 联合校验
- `HcomCheckOpParam(...)`（3 重载）— 综合校验：内部调 `HcomCheckTag` + `HcomCheckCount` + `HcomCheckDataType` + `CHK_PTR_NULL(stream)`

**B. legacy ascend950 层 V2 校验**（定义于 `src/legacy/ascend950/framework/entrance/param_check_v2.cc`）：

- `HcomCheckTagV2`、`HcomCheckCountV2`、`HcomCheckDataTypeV2`、`HcomCheckReductionOpV2`、`HcomCheckReduceDataTypeV2` — V2 版本的上述校验
- `HcomCheckOpParamV2`（5 重载）— V2 综合校验，内部同样调 `CHK_PTR_NULL(stream)` 等

**C. comm/buf/stream 指针校验 wrapper**：

- `CheckScatterInputPara(const HcclComm comm, const void* recvBuf)` — 检查 `CHK_PTR_NULL(comm)` + `CHK_PTR_NULL(recvBuf)`（`op_base.cc:2155`）
- `CheckChannelResParams(comm, channelDescs, channels, channelNum)` — 检查 comm/channelDescs/channels 指针 + channelNum 范围（`coll_comm_res_c_adpt.cc:739`）
- `ValidateThreadAcquireParams(engine/type/threadNum/config)` — engine/type/threadNum/config 校验（`thread_c_adpt.cc:120`）
- `CheckTeamCreateParam(comm/desc/team)` — comm/desc/team 校验（`hccl_team_c_adpt.cc:37`）
- `ValidateEndpointDesc(endpoint/endpointHandle)` — endpoint/endpointHandle 校验（`hcomm_endpoint_c_adpt.cc:99`）

> **注意**：以上 wrapper 函数清单为 HCOMM 仓实际存在的符号。旧版规则中引用的 `HcclCheckTag`、`CheckReduceScatterInputPara`、`CheckBroadcastInputPara`、`CheckAlltoAllInputPara`、`CheckReduceOp` 均为 HCCL 仓专属符号，在 HCOMM 仓中不存在，检视时勿引用。

**子规则C：公开API与内部函数区分（防止对内部函数过度要求）**：
- 公开API（声明在公开头文件中，用户直接调用）：必须检查 nullptr
- 内部函数（类的 private/protected 成员，仅被内部代码调用）：不强制要求（如果调用方保证非空）
- 回调函数（由框架调用）：必须检查

内部函数的指针参数豁免条件（满足任一即可豁免）：调用方通过 `std::make_unique`/`std::make_shared`/`new` 创建后直接传入；调用方传入对象引用的地址（`.get()`、`&member`）；调用方传入同函数内已校验过的指针；所有调用方都传入非空指针。

**子规则D：指针来源分析（防止对不可能为 nullptr 的指针报问题）**— 以下来源的指针永远不可能为 nullptr，不得报告为 nullptr 风险：

| 指针来源 | 为什么不可能为 nullptr |
|----------|----------------------|
| `char[]`/`T[]` 数组成员 | 数组是对象内联存储，取地址永远有效 |
| `std::string::c_str()` | C++ 标准保证返回非空指针 |
| `std::string::data()` | C++11 起保证返回非空指针 |
| `std::make_unique`/`std::make_shared` | 分配失败时抛 `std::bad_alloc` 异常，不返回 nullptr |
| `new`（非 nothrow） | 分配失败时抛异常 |
| C++ 引用 (`T&`) | 引用绑定后不可为空（标准保证） |
| `this` 指针 | 在成员函数中 `this` 永远有效 |
| 字符串字面量 | 编译期分配，永远有效 |

**子规则E：调用方上下文追踪（防止忽略调用方保证）**— 在报告内部函数的指针参数未检查 nullptr 前，必须追踪所有调用方，确认是否存在调用方传入 nullptr 的真实路径。如果所有调用方都保证传入非空指针，则不存在 nullptr 风险。

**误报防范检查清单**（报告指针问题前必须逐项确认）：
```
□ 子规则A — 代码事实验证：已读取 cited 源码，确认是指针使用/真正的 nullptr 检查/同一函数/函数签名一致
□ 子规则B — 跨函数校验追踪：已搜索首次使用前的校验调用，确认不存在等效 nullptr 校验
□ 子规则C — 公开API与内部函数区分：已判定函数类型，内部函数已确认调用方是否保证非空
□ 子规则D — 指针来源分析：已追踪指针实际来源，确认不属于"永远不可能为 nullptr"的类别
□ 子规则E — 调用方上下文追踪：已使用 git grep 找到所有调用方，确认存在调用方可能传入 nullptr 的真实路径
```

**任一检查项未通过 → 不得报告该问题。**

---

### 规则：工厂方法返回值与 dynamic_cast 结果必须做空指针检查后再使用

**风险等级**：**高风险（High）**（可能导致空指针解引用、进程崩溃、通信功能完全失效）

在 HCCL/Hcomm 代码中，大量使用工厂模式（Factory Pattern）创建算法操作对象，并通过 `dynamic_cast` 将基类指针转换为派生类指针以调用特定接口。此模式中存在两个独立的空指针风险点：

1. **工厂方法返回 nullptr**：工厂方法（如 `GetAlgOp`）可能因对象类型未注册、`new(std::nothrow)` 在极端内存压力下分配失败、工厂内部逻辑错误而返回 nullptr
2. **dynamic_cast 返回 nullptr**：`dynamic_cast<T*>(ptr)` 在 `ptr` 本身为 nullptr（C++标准保证 `dynamic_cast<T*>(nullptr) == nullptr`）或对象实际类型与目标类型不匹配时返回 nullptr

**核心风险**：当工厂方法返回 nullptr 且代码未做 nullptr 检查，直接对 `unique_ptr::get()` 返回的 nullptr 执行 `dynamic_cast`，结果仍为 nullptr，后续解引用导致 SIGSEGV 进程崩溃。此崩溃发生在函数内部，调用方的 nullptr 检查无法生效，因为函数根本不会正常返回。

**通用规则**：
- **规则 A**：工厂方法/注册表/`new(std::nothrow)` 创建对象后返回指针/智能指针的函数，返回值在使用前必须做空指针检查
- **规则 B**：`dynamic_cast<T*>(ptr)` 的返回值必须在解引用前做空指针检查。即使 `ptr` 不为 nullptr，`dynamic_cast` 在类型不匹配时也返回 nullptr。
- **规则 C**：如果通过当前调用逻辑可知dynamic_cast对象实际类型与目标类型匹配时，当 `ptr`必定不为nullptr，`dynamic_cast<T*>(ptr)` 的返回值可不做空指针检查。

| 检查项 | 检视方法 |
|--------|----------|
| 工厂方法/注册表返回值是否做 nullptr 检查 | 搜索 `GetAlgOp`、`Create`、`New`、`Instance().Get` 等工厂方法调用，确认返回值在使用前有 nullptr 检查 |
| `dynamic_cast` 结果是否做 nullptr 检查 | 搜索 `dynamic_cast<` 模式，确认 cast 后的指针在解引用前有 nullptr 检查 |
| `new(std::nothrow)` 返回值是否做 nullptr 检查 | 搜索 `new (std::nothrow)` 或 `new(std::nothrow)`，确认分配后立即检查返回值 |
| 调用方的 nullptr 检查是否能覆盖函数内部崩溃 | 评估函数内部是否有可能在返回前解引用 nullptr；若存在，则调用方检查无效，必须在函数内部添加保护 |
| ALLTOALL/ALLTOALLV/ALLTOALLVC 类型特殊处理分支 | 搜索 `dynamic_cast<AlltoAllOperator*>` 模式，确认所有使用点均有 nullptr 检查 |

**适用范围**：
- 算法操作对象创建：如 `CollAlgOpRegistry::GetAlgOp`、`HcclAlg::GetAlgOperator` 等工厂方法调用
- 通信操作对象创建：如 `Communicator` 创建、`Transport` 创建等
- dynamic_cast 类型转换：如 `dynamic_cast<AlltoAllOperator*>`、`dynamic_cast<AllReduceOperator*>` 等所有派生类转换
- 智能指针工厂返回值：如所有返回 `unique_ptr`/`shared_ptr` 的工厂方法
- nothrow new 分配：如所有使用 `new(std::nothrow)` 的对象创建

---

### 规则：公开 API 结构体指针成员使用前必须零初始化

**风险等级**：**低风险（Low）**（默认）→ **高风险（High）**（当且仅当当前代码存在确定可触发的野指针访问路径时升级）

HCCL/Hcomm 公开 API 的结构体中，指针成员如果未零初始化，当用户以栈分配方式创建实例时（如 `StructName info;`），C/C++ 不自动初始化局部变量，指针成员值为垃圾值（非 nullptr 的无效地址）。后续的 nullptr 检查（如 `CHK_PTR_NULL`）无法拦截垃圾值（垃圾值 ≠ nullptr），检查通过后解引用垃圾指针导致段错误（SIGSEGV）。C++ 的"部分初始化"规则不保证未显式初始化的成员被清零。

**C 兼容性约束**（强制执行）：

`include/` 下的公开 API 头文件需被 C 代码 `#include`（纯 C 头）。C 标准不允许在结构体定义内写成员默认初值（`= nullptr` / `= NULL` 均非法），`nullptr` 本身也是 C++11 关键字。因此**禁止在 C 兼容头文件的结构体定义中写 `= nullptr` 成员默认初值**，否则会破坏 `include/` 头文件的 C 兼容性。按头文件语言兼容性分两种策略落实"指针成员使用前零初始化"：

| 头文件类型 | 结构体定义中的写法 | 使用前的零初始化方式 | 示例 |
|------------|------------------|---------------------|------|
| **C++ 专用头**（仅被 .cc/.cpp include，不暴露给 C） | 允许在结构体定义中写 `= nullptr` 成员默认初值 | 定义即初始化，无需使用方额外操作 | `struct Foo { int* p = nullptr; };` |
| **C 兼容头**（`include/` 下需被 C 代码 include 的头文件） | **禁止**在结构体定义中写 `= nullptr`/`= NULL` 成员默认初值 | 使用方在栈分配实例时必须用 `{0}` 整体初始化或 `memset(&info, 0, sizeof(info))` 清零 | `Foo info{0};` 或 `memset(&info, 0, sizeof(info));` |

> **判定头文件语言兼容性的方法**：检查头文件是否包含 `extern "C"` 包裹、是否被 `.c` 文件 `#include`、是否位于 `include/` 对外目录。满足任一即视为 C 兼容头，适用 C 兼容头策略。

**风险等级判定规则**（强制执行）：
- **默认低风险**：结构体指针成员未零初始化，仅存在潜在风险，默认判定为低风险
- **高风险升级条件**：当且仅当满足以下全部条件时升级：(1) 当前代码存在确定的执行路径访问该未初始化指针成员；(2) 可被实际触发；(3) 访问时为垃圾值且 nullptr 检查无法拦截

| 检查项 | 检视方法 |
|--------|----------|
| C 兼容头中结构体是否误写 `= nullptr`/`= NULL` 成员默认初值 | 搜索 `include/` 下头文件的 `typedef struct` / `struct` 定义，确认 C 兼容头的结构体定义内**未出现** `= nullptr` / `= NULL` 成员默认初值（出现即为违规，破坏 C 兼容性） |
| C++ 专用头中结构体指针成员是否有 `= nullptr` 默认初值 | 搜索 C++ 专用头中的 `struct` 定义，确认每个指针成员有 `= nullptr` |
| C 兼容头结构体的栈分配实例使用前是否零初始化 | 搜索 C 兼容头结构体类型的栈变量声明，确认使用 `{0}` 初始化或在使用前 `memset(&info, 0, sizeof(info))` |
| 非公开结构体指针成员初始化 | 内部结构体也应遵循，但可放宽至构造函数初始化或 `{0}` 整体初始化 |
| 仅部分成员有默认初值的结构体（仅 C++ 专用头） | 搜索 `= 0` / `= nullptr` 在结构体定义中出现但其他指针成员无初值的场景，补全所有指针成员的默认初值 |
| 风险等级判定 | 默认低风险；追踪使用点，仅当当前代码存在确定触发路径时升级为高风险 |

---

### 规则：内存分配函数返回值必须做空指针检查，free/delete 后指针必须立即赋予新值

**风险等级**：**高风险（High）**（可能导致空指针解引用、悬挂指针 double-free、进程崩溃）

**分配后检查缺失**：`malloc`/`new(std::nothrow)` 等分配函数在内存不足时返回 nullptr，若不做检查直接使用返回指针，后续解引用导致 SIGSEGV。

**释放后悬挂指针**：`free`/`delete` 后未将指针置 nullptr（或新值），指针变为悬挂指针（dangling pointer）。后续再次引用该指针可能导致 double-free 或 use-after-free。

| 检查项 | 检视方法 |
|--------|----------|
| malloc 后是否有 NULL / nullptr 检查 | 搜索 `malloc`、`calloc`、`realloc` 调用，确认返回值在使用前有 `if (ptr == NULL)` 或 `CHK_PTR_NULL` 检查 |
| `new(std::nothrow)` 后是否有 nullptr 检查 | 搜索 `new (std::nothrow)` 或 `new(std::nothrow)`，确认分配后立即检查返回值 |
| 普通 new 后的 nullptr 检查是否为死代码 | 搜索 `new`（非 nothrow）后接 `if (ptr == nullptr)` 的模式，普通 new 失败抛 std::bad_alloc 而非返回 nullptr，nullptr 检查为死代码 |
| free/delete 后指针是否置 nullptr | 搜索 `free`、`delete` 调用，确认释放后立即将指针置 nullptr（或离开作用域） |
| free 结构体成员后是否将成员置 nullptr | 搜索 `free(structPtr->member)` 模式，确认释放后将 `structPtr->member = nullptr` |
| 函数返回值可能为 NULL 的调用是否做检查 | 搜索返回 `char*`/`void*` 等指针类型的函数调用（如 `RankToString`、`malloc` 包装函数），确认返回值在使用前做 NULL 检查 |

**HCCL/Hcomm 项目中的统一错误处理模式**：

| 宏 | 用途 | 正确用法 |
|-----|------|----------|
| `CHK_PTR_NULL(ptr)` | 检查指针是否为 nullptr，若为 nullptr 则返回 `HCCL_E_PTR` | 用于公共 API 参数检查 |
| `CHK_SMART_PTR_NULL(smartPtr)` | 检查智能指针是否为 nullptr，若为 nullptr 则返回 `HCCL_E_PTR` | 用于 unique_ptr/shared_ptr 返回值检查 |
| `CHK_PRT_RET(condition, msg, retCode)` | 检查条件，若为 true 则打印错误并返回指定错误码 | 用于自定义条件检查 |

> **注意**：`CHK_PTR_NULL` 和 `CHK_SMART_PTR_NULL` 仅检查 `nullptr`，无法拦截垃圾指针值（非 nullptr 的无效地址）。因此，公开 API 结构体的指针成员使用前必须零初始化：C++ 专用头可在定义中写 `= nullptr` 默认初值；C 兼容头禁止写 `= nullptr`/`= NULL`，须由使用方以 `{0}` 整体初始化或 `memset(&info, 0, sizeof(info))` 兜底（参见「公开 API 结构体指针成员使用前必须零初始化」规则）。

**适用范围**：
- malloc/calloc/realloc 分配：所有 C 代码中的动态内存分配
- new/new(std::nothrow) 分配：所有 C++ 的对象创建
- 工厂方法返回值：如 `CollAlgOpRegistry::GetAlgOp`、`RankToString` 等返回指针/智能指针的函数
- free/delete 释放：所有动态内存释放操作
- HCCL_MEM_TYPE_DEVICE 分配：如 `hrtMalloc`、`DeviceMem::alloc` 等设备内存分配同样需检查返回值, 保证申请的内存非空

---

## 数值计算安全

### 规则：除法/取模运算前必须校验除数不为零

**风险等级**：**致命（Critical）**（整数除零为未定义行为，典型表现为 SIGFPE 致进程崩溃）

C/C++ 标准中，整数除法 `/` 与取模 `%` 的除数为 0 属未定义行为（Undefined Behavior）。在主流平台上触发硬件异常（x86 触发 `#DE`，Linux 向进程投递 `SIGFPE`），导致进程崩溃。浮点除零不触发异常但产生 `inf`/`nan`，后续运算污染数据。

HCCL/Hcomm 项目中，除数常来源于配置参数、拓扑解析结果、结构体字段等，这些值可能因初始化缺失、解析失败残留、异常输入而取 0。若运算前未校验，即存在真实除零触发路径。

**关键认知**：
- 结构体字段的默认初值常为 0（如 `u32 serverNum { 0 };`），"指针已判空/对象已构造"不能保证数值字段非 0
- 指针有效（如 `pComm != nullptr`）与数值字段有效（如 `serverNum != 0`）是两个独立的不变量，前者成立不蕴含后者成立
- 仅记录日志而不返回错误的"软校验"无法阻止后续除零运算

| 检查项 | 检视方法 |
|--------|----------|
| `/`、`%` 运算的除数是否校验非 0 | 搜索 `/`、`%` 运算符，追踪除数来源：若是变量/字段/返回值，确认运算前有 `if (x == 0)` 或 `CHK_PRT_RET(x == 0, ...)` 校验 |
| 除数来源于结构体字段时是否依赖默认初值 | 查看结构体定义，若字段默认 `0`/`{0}`，确认使用前有显式非 0 校验，不得依赖"对象已构造" |
| 除数来源于配置/拓扑解析时是否校验 | 追踪除数赋值链，若来自 `rankTable`、配置文件、网络数据，确认解析阶段或使用前校验非 0 |
| 异常分支是否"仅日志不返回" | 检查 `if (异常条件) { HCCL_ERROR(...); }` 后是否紧跟 `return` 错误码；若继续执行后续运算，判定为错误吞没 |
| 浮点除零 | 浮点 `/0.0` 产生 `inf`/`nan` 不崩溃但污染数据，需校验除数或检查结果 `isinf`/`isnan` |
| 复合赋值除法 | `a /= b`、`a %= b` 同样需校验 `b` 非 0 |

**适用范围**：
- 整数除法/取模：`a / b`、`a % b`，`b` 为变量/字段/返回值时必须校验非 0
- 浮点除法：`a / f`，`f` 为变量时建议校验非 0 或检查结果
- 复合赋值：`a /= b`、`a %= b`，同样需校验 `b`
- 隐式除法：如 `平均值 = 总数 / 数量`、`每服务器设备数 = 总设备数 / 服务器数` 等拓扑/统计计算
- 除数来源为结构体字段：尤其关注 `RankTable_t::serverNum`、`deviceNum`、`rankNum` 等默认初值为 0 的字段
- 除数来源为配置/解析结果：拓扑解析、rank table 解析、配置文件读取得到的数值，使用前必须校验

---

### 规则：整数运算结果用于高危场景必须校验合法性

**风险等级**：**致命（Critical）/ 高风险（High）**（视用途场景，可能导致缓冲区溢出、越界访问、内存破坏、死循环等）

整数运算（算术运算、位运算、类型转换等）可能产生溢出、截断、翻转、除零等异常结果。整数运算之后，只要将计算结果用于下列任一高危用途，都必须在使用前添加校验。

**七种高危用途（必须校验）**：

| 序号 | 用途 | 异常后果 | 校验内容 |
|------|------|----------|----------|
| 1 | 作为数组索引 | 越界读写、缓冲区溢出 | 索引 ≥ 0 且 < 数组长度；负索引/超大索引拦截 |
| 2 | 作为指针运算偏移 | 越界指针、非法内存访问 | 偏移后指针仍在合法对象范围内 |
| 3 | 作为对象的长度或大小 | 长度错误致读写越界、缓冲区溢出 | 长度 ≤ 实际分配大小；非负；不超上限 |
| 4 | 作为数组/缓冲区边界 | 边界错误致越界 | 边界 ≤ 实际容量 |
| 5 | 作为内存分配函数的实参 | 分配 0 字节/超大分配/整数溢出致分配不足 | 分配大小 > 0；不溢出；≤ 上限 |
| 6 | 作为循环终止条件 | 死循环、无限循环、跳过循环 | 循环变量递增/递减方向正确；终止条件可达 |
| 7 | 作为拷贝长度 | 拷贝越界、缓冲区溢出 | 拷贝长度 ≤ 源/目标缓冲区容量 |

**真实可达路径约束（强制）**：报告异常时必须有真实可达的触发路径，不得仅凭"理论上可能"报告问题。判定逻辑：
- 当前代码在当前运行条件下存在真实的触发路径和触发条件 → 按实际影响判定风险等级
- 当前代码本身不会出问题，仅因后续修改/新增代码才可能触发 → 判定为低风险，标注"潜在风险"
- 无法构造出具体触发条件 → 不报告该问题

| 检查项 | 检视方法 |
|--------|----------|
| 整数运算结果是否用于数组索引 | 搜索 `array[expr]`、`ptr[expr]` 模式，若 `expr` 含运算，校验结果范围 |
| 整数运算结果是否用于指针运算 | 搜索 `ptr + expr`、`ptr += expr`、`ptr - expr`，校验偏移后指针合法 |
| 整数运算结果是否作为长度/大小 | 搜索 `memcpy(dst, src, expr)`、`malloc(expr)`、`new T[expr]`，校验 expr 非负且不溢出 |
| 整数运算结果是否作为数组边界 | 搜索 `for (i = 0; i < expr; i++)`、`if (idx < expr)`，校验边界 ≤ 容量 |
| 整数运算结果是否作为内存分配实参 | 搜索 `malloc/calloc/realloc/new` 的 size 参数，若含运算（如 `a * b`），校验不溢出 |
| 整数运算结果是否作为循环终止条件 | 搜索 `for/while` 的条件，若含运算，校验循环可达终止 |
| 整数运算结果是否作为拷贝长度 | 搜索 `memcpy/memmove/strncpy/read/write` 的 len 参数，校验 ≤ 缓冲区容量 |
| 宽窄类型转换是否截断 | 搜索 `u64 → u32`、`int → char` 等隐式/显式转换，校验值在目标范围 |
| 有符号整数运算是否溢出 | 有符号整数溢出为 UB，搜索 `a + b`、`a * b`，若可能溢出，用安全函数或校验 |
| 报告问题是否有真实可达路径 | 每个报告的问题必须给出数据来源、触发条件、影响路径；无法构造触发条件的不报告 |

**适用范围**：
- 算术运算：`+`、`-`、`*`、`/`、`%`
- 位运算：`<<`、`>>`（左移可能溢出/改变符号位）
- 类型转换：宽类型→窄类型（截断）、有符号↔无符号（翻转）
- 自增自减：`++`、`--`（可能溢出）
- 复合赋值：`+=`、`-=`、`*=`、`/=`、`%=`、`<<=`、`>>=`
- 库函数返回值：`strlen`、`sizeof` 结果参与运算后用于高危用途

---

## 命令注入与路径穿越防护

### 规则：禁止将外部输入未经校验作为命令执行函数的参数

**风险等级**：**致命（Critical）**（命令注入可导致任意代码执行、权限提升、数据泄露、系统被完全控制）

程序执行由外部输入构建的全部或部分命令语句时，若外部输入未经严格校验，攻击者可通过构造特殊输入注入额外命令或修改原命令语义，导致实际执行结果与预期不一致，产生命令注入漏洞。

**外部输入来源**（均不可信）：用户输入（命令行参数、交互输入、API 请求参数）、接口数据报文（网络报文、RPC 请求、消息队列内容）、操作系统环境变量（`getenv` 返回值）、普通用户可修改的配置文件。

**常见注入手法**：命令分隔符（`;`、`|`、`||`、`&&`、`&`）、命令替换（`` ` ``、`$()`）、重定向（`>`、`>>`、`<`）、转义/引号逃逸（`'`、`"`、`\`）、路径穿越（`../`、绝对路径覆盖）。

**关于 `open()`**：`open()` 本身不执行 shell 命令，不属于命令注入；但其路径参数若来自外部输入未经校验，可能导致任意文件读写/覆盖/路径穿越。若 `open()` 打开的脚本文件后续被 shell 执行，则间接形成命令注入链路。故 `open()` 的路径参数同样必须校验。

**安全替代方案**：

| 危险函数 | 安全替代 | 说明 |
|----------|----------|------|
| `system(cmd)` | `fork()` + `execvp()` | 不经过 shell，参数以数组传递，无元字符解析 |
| `popen(cmd, "r")` | `pipe()` + `fork()` + `execvp()` | 同上 |
| `snprintf` 拼接 + `system` | 白名单校验 + `execvp` | 必须拼接时严格白名单 |
| `getenv` + 命令执行 | 校验环境变量值 + `execvp` | 环境变量不可信，需校验 |
| `open(外部路径)` | 路径规范化 + 白名单目录校验 | 防止路径穿越，限制在允许目录内 |

> **核心原则**：优先使用不经过 shell 的 `exec*` 系列函数（`execvp`/`execv`），以参数数组形式传递，从根本上避免元字符注入。必须用 shell 时，对输入做严格白名单校验。`open()` 的路径参数必须校验路径穿越。

| 检查项 | 检视方法 |
|--------|----------|
| `system()` 参数是否含外部输入 | 搜索 `system(` 调用，追踪参数来源：若含用户输入/报文/环境变量/配置文件，判定为风险 |
| `popen()` 参数是否含外部输入 | 搜索 `popen(` 调用，同上 |
| `execl`/`execlp`/`execv`/`execvp` 是否经 shell | `execvp`/`execv` 不经 shell（参数数组，安全）；若传 `/bin/sh -c` 仍可注入 |
| 环境变量是否用于命令构造 | 搜索 `getenv()` 返回值流入 `system`/`popen`/`exec` 参数 |
| 配置文件内容是否用于命令构造 | 追踪配置读取结果是否流入命令执行函数 |
| `snprintf`/`sprintf` 拼接命令字符串 | 搜索 `snprintf(cmd, ..., "...%s...", input)` + `system(cmd)` 模式 |
| 是否使用白名单校验 | 外部输入用于命令时，必须有白名单校验（仅允许安全字符集），黑名单不可靠 |
| `open()` 路径参数是否来自外部输入 | 搜索 `open(` 调用，若路径含外部输入，校验路径穿越（`../`、绝对路径） |
| `open()` 打开的文件是否后续被执行 | 若 `open()` 写入脚本文件后被 shell 执行，形成间接命令注入链路 |

**适用范围**：
- 命令执行函数：`system()`、`popen()`、`execl()`/`execlp()`/`execle()`/`execv()`/`execvp()`/`execvpe()`
- 外部输入来源：用户输入、接口数据报文、环境变量（`getenv`）、普通用户可修改的配置文件
- 命令拼接场景：`snprintf`/`sprintf`/`strcpy` 拼接命令字符串后传入 `system`/`popen`
- 间接执行：`open()` 写入脚本文件后通过 shell 执行、调用其他程序的 `-e`/`--exec` 选项
- 文件打开场景：`open()`/`fopen()` 的路径参数来自外部输入时，必须校验路径穿越
- HCCL/Hcomm 项目：诊断命令、工具调用、脚本执行、配置文件读取等场景

---

### 规则：禁止外部输入未经校验作为文件路径，防止路径穿越

**风险等级**：**高风险（High）**（路径穿越可导致任意文件读/写/覆盖，敏感信息泄露或系统文件被篡改）

路径穿越攻击（Path Traversal）是一种通过操纵文件路径来访问或执行存储在 Web 文档根目录之外的文件和命令的攻击技术。攻击者通常利用特殊字符序列（如 `../` 或绝对路径）来绕过目录限制，从而访问敏感文件或系统资源。包括相对路径穿越（通过 `../` 访问上级目录）和绝对路径穿越（直接指定文件的绝对路径）。

| 检查项 | 检视方法 |
|--------|----------|
| `open()`/`fopen()` 路径参数是否来自外部输入 | 搜索 `open(`/`fopen(` 调用，追踪路径参数来源：若含用户输入/报文/环境变量/配置文件，判定为风险 |
| 路径中是否包含 `../` 序列 | 检查路径拼接结果是否含 `../`，存在则可穿越上级目录 |
| 路径是否为外部指定的绝对路径 | 检查路径是否以 `/` 开头且来自外部输入，存在绝对路径穿越风险 |
| 路径是否经过规范化与白名单目录校验 | 确认路径经 `realpath()` 规范化后位于允许目录内 |

---

## 拒绝服务防护

### 规则：外部输入必须进行合法性校验，防止拒绝服务

**风险等级**：**致命（Critical）/ 高风险（High）**（可导致系统宕机、复位、CPU/内存资源过度消耗、服务或进程不可用）

系统使用来自用户输入、通信协议、用户态输入、第三方系统、外部接口等外部输入数据时，未正确进行合法性校验，可能导致系统宕机、复位、CPU/内存资源过度消耗，从而引发拒绝服务（DoS）、服务或进程不可用。

**外部输入来源**（均不可信）：用户输入、通信协议报文、用户态输入、第三方系统数据、外部接口输入数据。

**常见拒绝服务场景**：
1. **内存泄露型拒绝服务**：外部输入数据未作合法性校验或校验不正确，引起内存泄露，导致系统资源耗尽
2. **死循环型拒绝服务**：使用外部协议报文作为循环输入条件，或作为循环输入条件的判断依据，导致死循环，CPU 资源过度消耗
3. **超大内存申请型拒绝服务**：外部数据作为内存大小参数，导致可申请超大内存，系统资源耗尽拒绝服务
4. **反序列化未校验型拒绝服务**：从远端报文反序列化得到的字段（如数组长度/元素个数）未经校验直接用于 `resize`/循环计数，攻击者构造恶意超大值导致内存耗尽或循环读取脏数据

**防护方案**：

| 防护措施 | 说明 | 适用情况 |
|----------|------|----------|
| 外部输入合法性校验 | 对外部输入做白名单校验（类型/格式/长度/范围），拒绝非法输入 | 所有情况 |
| 异常路径资源释放 | 错误处理分支使用 `goto cleanup` 或 RAII，确保释放已分配资源 | 内存泄露 |
| 循环上限校验 | 外部报文作为循环条件时，设最大迭代次数，超限中断 | 死循环 |
| 内存大小上限校验 | 外部数据作为 size 时，校验 ≤ 合理上限，拒绝超大申请 | 超大内存 |
| 反序列化字段前置校验 | 反序列化数值字段在用于 `resize`/循环前，校验与期望值一致性及上界，一致性校验不得在 `resize` 之后 | 反序列化未校验 |
| 资源配额限制 | 对每用户/每连接/每进程的资源使用设配额，防止资源耗尽 | 通用 |
| 超时机制 | 处理外部请求设超时，超时中断，防止 CPU 过度消耗 | 通用 |

> **核心原则**：所有外部输入均不可信，使用前必须做合法性校验。重点防护四类拒绝服务：内存泄露（异常路径释放资源）、死循环（循环上限校验）、超大内存申请（size 上限校验）、反序列化未校验（反序列化字段在 resize/循环前校验，一致性校验不得在 resize 之后）。

| 检查项 | 检视方法 |
|--------|----------|
| 外部输入是否做合法性校验 | 追踪用户输入/通信协议/用户态/第三方/外部接口数据，确认使用前有校验（类型/格式/长度/范围） |
| 异常路径是否释放内存 | 检查错误处理分支（`return`/`break`/`continue`）是否释放已分配内存/关闭资源句柄 |
| 外部报文是否作为循环条件 | 搜索 `while`/`for` 循环，若条件或计数器来自外部报文（`length`/`count`/`size`），校验上限和递减逻辑 |
| 循环是否可达终止 | 分析循环终止条件，确认外部输入不会导致无法终止（死循环） |
| 外部数据是否作为内存大小参数 | 搜索 `malloc`/`calloc`/`realloc`/`new` 的 size 参数，若来自外部输入，校验上限 |
| 内存申请是否有上限校验 | 外部数据作为 size 时，必须有最大值校验，防止申请超大内存 |
| 反序列化字段是否校验后再用于 resize/循环 | 搜索 `>>` / `deserialize` / `read` 结果流入 `resize(`/循环计数，确认有与期望值一致性和上界校验 |
| 资源分配是否有配额限制 | 重复处理外部请求时，是否有资源配额（连接数/内存总量/处理速率）限制 |

**适用范围**：
- 外部输入来源：用户输入、通信协议报文、用户态输入、第三方系统数据、外部接口输入
- 内存泄露场景：外部输入触发异常路径未释放资源
- 循环场景：外部报文作为循环条件或判断依据
- 内存申请场景：外部数据作为 `malloc`/`calloc`/`realloc`/`new` 的 size 参数
- HCCL/Hcomm 项目：通信协议报文处理、用户态 API 入参、第三方系统交互等场景

---

## 环境变量校验

本章规范环境变量读取数据的安全校验。环境变量可被普通用户任意修改，属于不可信外部输入，必须按数据用途进行针对性校验。

### 规则：从环境变量读取的数据必须视为外部数据并分类校验

**风险等级**：**高风险（High）**（环境变量未校验可导致数值截断、路径穿越、数组越界、系统被控制）

对于从环境变量读取的数据，全部认为是外部数据（不可信）。必须根据数据用途进行分类校验：
- **数值类型**：需要进行数据截断、数据非法等拦截。环境变量以字符串形式存储，转换为数值时可能发生截断（如 `int` 溢出）、非法字符（非数字内容）等问题，必须校验转换结果的有效性和范围
- **路径类型**：需要进行路径合法性校验。环境变量作为路径使用时，可能包含 `../` 路径穿越、绝对路径越权访问等风险，必须校验路径规范化和白名单目录
- **数值且作为数组下标**：需要进行范围校验，防止数组溢出。环境变量转换为数值后用作数组索引时，若未校验范围，可能导致负数下标或超长下标，引发越界读写

| 检查项 | 检视方法 |
|--------|----------|
| `getenv()` 返回值是否作为外部数据校验 | 搜索 `getenv(` 调用，确认返回值经过校验后才使用 |
| 环境变量转为数值是否校验截断与非法值 | 检查 `atoi`/`strtol`/`strtoul` 等转换结果，确认转换成功且值在预期范围内 |
| 环境变量作为路径是否校验路径合法性 | 检查环境变量作为文件路径使用时，是否经 `realpath()` 规范化并校验位于允许目录内 |
| 环境变量作为数组下标是否校验范围 | 检查环境变量转换数值后用作数组索引时，是否校验 ≥ 0 且 < 数组长度，防止越界 |
| 环境变量是否直接用于命令执行或内存分配 | 追踪 `getenv()` 返回值是否流入 `system`/`open`/`malloc` 等，确认已分类校验 |

**适用范围**：
- 数值场景：环境变量转为数值用于配置参数、循环次数、缓冲区大小等
- 路径场景：环境变量作为文件路径、目录路径、库搜索路径等
- 数组下标场景：环境变量转为数值后用作数组索引访问
- HCCL/Hcomm 项目：日志路径、通信端口、设备 ID、调试开关等环境变量配置场景

---

## 并发安全

- 全局/静态可变状态必须有锁、原子或屏障保护；检视时 grep 所有引用点确认访问线程
- 持锁期间禁止调用可阻塞的同步操作（如长时间 `Synchronize`），易死锁/饿死
- `thread_local`、不可变对象天然线程安全，可作误报排除依据

---

## 错误处理

- 错误码语义准确：不得用 EAGAIN 表达"已存在"等易误导语义
- 系统调用/安全函数返回值必须检查；失败路径资源回滚完整
- 异常安全：`delete`/`move`/`swap` 建议标 `noexcept`；catch(...) 不吞错误

---

## 工具验证纪律（防误报）

对以下发现必须先工具验证再报告（读函数签名/实现、grep 调用方）：

- 指针赋值/传递（值传递 vs 引用）
- 算术溢出（确认类型与值域；枚举值、循环计数器等值域受限场景可排除）
- 结构体成员增删（grep 成员名检查所有引用点同步修改）
- 跨文件遗漏（grep 被修改/删除的标识符的全部引用）

反证检验：报告前反问"什么条件下该问题不成立"，用工具验证该条件（如调用链上游已判空、RAII 已管理该资源）；反证成立则不报。

### 风险等级参考

- **致命（Critical）**: 可能导致系统崩溃、数据丢失、安全漏洞被利用、大规模资源泄漏、核心功能完全不可用。此类问题一旦触发，影响不可恢复或恢复代价极高
- **高风险（High）**: 可能导致资源泄漏、局部崩溃、安全漏洞、数据损坏
- **中风险（Medium）**: 可能导致功能异常、性能下降、兼容性问题
- **低风险（Low）**: 代码质量问题、可读性问题、格式问题

### 阻塞性问题判定

存在以下任一情况为阻塞性问题：
- 资源泄漏风险
- 安全漏洞
- 逻辑错误
- 违反华为编码规范（使用禁止的构造、命名严重违规、内存管理不当）
- 违反华为安全编码规范（使用危险函数、整数溢出、缓冲区溢出）
- 违反 Hcomm 项目编码规范（未初始化结构体、错误处理不符合规范、敏感信息泄露）
- 缺少必要的错误处理
