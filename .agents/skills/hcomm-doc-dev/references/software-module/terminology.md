# 中英术语对照与翻译句式

英文版（`_en.md`）与中文版逐节语义对齐，非逐句直译。代码符号（类名/函数名/常量/错误码/十六进制值）原样保留。

## 章节标题对照

| 中文 | 英文 |
|------|------|
| `<模块名> 模块代码解析` | `<module> Module Code Analysis` |
| `<模块名> 功能说明`（存量 H1 变体，触碰更新时统一为"模块代码解析"） | `<Module> Function Description`（存量变体，触碰更新时统一为 Module Code Analysis） |
| 功能描述 | 软件特性模块用 Feature Description；针对单一接口/函数的文档用 Function Description（存量触碰更新时按模块类型归位） |
| 对外接口 | External Interfaces |
| 目录描述 | Directory Description |
| 文件间关系 | File Relationships |
| `<X>`文件交互 | `<X> File Interaction` |
| 流程描述 | Flow Description |
| 数据描述 | Data Description |
| 接口描述 | Interface Description |
| 公共接口 | Public Interfaces / Public API |
| 内部 / 私有接口 | Internal / Private Interfaces |
| 使用限制 | Usage Limitations |
| 支持的场景 | Supported Scenarios |
| 规格约束 | Constraint Specifications / Specifications |
| 已知限制 | Known Limitations |
| 使用场景 | Usage Scenarios / Application Scenarios |
| 与周边模块的关系 | Relationships with Surrounding Modules |
| 依赖 | depends on / dependency |
| 配置方法（存量扩展章节变体） | Configuration |
| 调用周期（存量扩展章节变体） | Call Cycle |
| 监控日志格式（存量扩展章节变体） | Monitoring Log Format |

> **标题统一规则（B5）**：英文章节标题一律用本表标准译名，不用 Overview/API/Constraints 等变体；存量英文版（如 streamTaskMonitor/README_en.md 的 Overview、API、Constraints）触碰更新时随骨架调整统一为标准译名（需用户确认）。
>
> **章节名统一裸标题（C4）**：章节名不加括号后缀。存量变体"流程描述（Mermaid时序图）""数据描述（Mermaid类图）""接口描述（类图）"触碰更新时建议调整为裸标题（需用户确认）。
>
> **存量扩展章节（C3）**：配置方法/调用周期/监控日志格式/模块组成与职责分工等存量扩展章节不承认为标准章节，触碰更新时建议并入标准章节或随骨架调整处理（需用户确认）；上表末三行译名仅供参考。

## 常用技术术语

| 中文 | 英文 | 备注 |
|------|------|------|
| 集合通信库 | collective communication library | |
| 通信域 | communicator | 与仓内代码一致（communicator 目录、CollComm 类） |
| Host 侧 / Device 侧 | Host side / Device side | 保留原文，不译。Device 侧按实际执行单元写实名（AICPU/CCU/MC2 等）；存量文档中的"Host 侧 / AICPU 侧"为历史变体，触碰更新时随 Device 侧口径统一（需用户确认） |
| 算子 | operator | |
| 单例 | singleton | |
| 守护线程 | daemon thread | |
| 回调 | callback | |
| 注册/注销 | register / unregister | |
| 上报 | report | |
| 前序任务上下文 | preceding task context | |
| 错误码转换 | error code conversion | |
| 规格约束 | constraint specifications | |
| 线程安全 | thread safety | |
| 空指针保护 | null pointer protection | |
| 弱符号 | weak symbol | `__attribute__((weak))` 原样保留 |
| 动态加载 | dynamic loading | dlopen 语境 |

## 缩写处理

中英两版首次出现都要注释全称：

- 中文版：`CQE（Completion Queue Entry，完成队列项）`
- 英文版：`CQE (Completion Queue Entry)`，后文直接用缩写
- 例外：业界通用缩写（API、SDK）可不注

## 翻译句式对照

| 中文句式 | 英文句式 |
|---------|---------|
| 负责在……时进行…… | is responsible for ... when ... |
| 核心能力包括： | Core capabilities include: |
| 该模块横跨 Host 侧和 Device 侧两个运行环境 | This module spans two runtime environments: the Host side and the Device side |
| 以守护线程方式定期检测…… | Runs as a daemon thread that periodically checks for ... |
| 通过……通道上报到…… | reports to ... through the ... channel |
| 最大支持 N 个设备 | supports a maximum of N devices |
| 超出时打印警告 | prints a warning when exceeded |
| 防止同一……重复上报 | prevents duplicate reporting of the same ... |
| 使用场景：<条件/流程>时，由 <上游调用方> 调用本模块 | Usage scenario: when <condition/process>, this module is invoked by <upstream caller> |
| 依赖……获取…… | depends on ... for obtaining ... |
| 由……持有，按……访问 | held by ... and accessed via ... |
| 本模块与……通过……交互 | this module interacts with ... through ... |

## 结构件翻译

- 目录树注释：`# Host侧异常处理` → `# Host-side exception handling`
- 表头：文件/功能/依赖关系 → File / Function / Dependencies；接口/类型/参数/返回值/功能说明 → Interface / Type / Parameters / Return Value / Description
- 表内类型：公有（对外） → public (external)；公有静态 → public static；私有 → private；`[in]`/`[out]` 保留
- Mermaid 图内文字：节点名/函数名保留原文，中文注释（Note、subgraph 名、边标签短语）需译成英文。例：`Note right of HCD: 更新开关状态` → `Note right of HCD: Update switch status`；subgraph `Host侧` → `Host Side`
- `✅`/`❌` 保留

## hcomm 硬件代号规避（强制）

来源：hcomm 仓 `.agents/skills/hcomm-review/references/markdown-writing.md` 的"硬件代号规避"规则，对本文档输出全过程生效（中英两版均适用）：

- **md 文档正文不使用内部硬件代号**：CKE、Xn、GSA、MS；统一替换为 C++ API 类名。
- **例外**：代码块内、反引号包裹的代码标识符（如源码中的 `g_communicatorCallbackMapV2`、函数名）不受此限——规避的是"正文叙述用硬件代号指代机制"，不是源码符号本身。
- **既存文档处理**：改写涉及旧代号的内容时（如"检查 CKE 期望值"），改用 API 类名表述（"检查 Event 期望值"）。

| 内部硬件代号（禁用于正文） | 替换为 C++ API 类名 |
|--------------------------|-------------------|
| CKE | Event |
| Xn | Variable |
| GSA | Address |
| MS | CcuBuffer |

## 英文版自查

1. 无翻译腔：不出现 "the the"、直译的"进行……操作"（do ... operation）。
2. 章节结构与中文版完全一致（标题数、表数、图数）。
3. 代码符号、常量值、错误码值与中文版逐一对应。
