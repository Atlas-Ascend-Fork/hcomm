---
name: hcomm-review
description: 检视本仓（hcomm）在 GitCode 上的 PR。触发词：检视、检视PR、检视本仓PR、代码检视、代码审查、review PR、code review、检视PR123、review hcomm PR。从 GitCode 拉取指定 PR 的代码到隔离 worktree，多维度检视（代码质量/风格/逻辑/注释/架构/AGENTS.md 合规/描述与实现吻合度/功能正确性/文档质量/模块目录文档同步），把检视意见按行提交到 PR（自动行号验证、与已有评论去重），提交检视汇总报告，最后清理 worktree 与临时文件。支持 Windows 与 Linux。
---

# hcomm PR 代码检视

从 GitCode 拉取指定 PR 的代码到隔离 worktree，多维度检视，按行提交检视意见，提交汇总报告，清理现场。任意给一个 PR 即可检视。全程全自动执行，不要中途询问用户。检视意见统一用中文（代码标识符、severity 级别名保留英文）。

## 首次使用配置

```bash
# 1) GitCode token（二选一）:
#    a) 环境变量: export GITCODE_TOKEN=<token>  (Windows: $env:GITCODE_TOKEN 或 set)
#    b) git credential 自动读取（已 git clone 凭据时无需配置）
# 2) 本仓本地 clone 路径: 默认当前目录，或用 --repo-root 指定

# 配置校验（通过后无需再配）
python3 .agents/skills/hcomm-review/scripts/pr_review.py --check-env --repo-root <本仓路径>
```

依赖：Python 3.7+（仅标准库）、git、能访问 gitcode.com。Windows 与 Linux 均可运行。

## 检视流程（五步）

用户说"检视 PR123"（或给 PR 链接）时，解析 PR 编号后按 Step 1→5 顺序执行。

### Step 1 准备基线与 worktree

- 取 PR 元数据（state、目标分支、base/head sha、变更文件数）：

```bash
python3 .agents/skills/hcomm-review/scripts/pr_review.py --pr <N> --meta --repo-root <本仓路径>
```

- **确认目标分支**：元数据的 `target_branch` 是 `master` 还是其他分支。worktree 基于该分支的基线创建：

```bash
cd <本仓路径>
git fetch <remote> <target_branch>                    # remote 用 origin 或 upstream（指向 cann/hcomm 的那个）
git fetch <remote> refs/merge-requests/<N>/head:refs/remotes/<remote>/mr/<N>-head
# <临时目录> 用系统临时目录（Linux /tmp、Windows %TEMP%），不放仓内，避免 worktree 出现在 git status
git worktree add --detach <临时目录>/review-<N> refs/remotes/<remote>/mr/<N>-head
```

- **基线核验**（防基线漂移把意见发到非本 PR 的代码上）：本地 `git diff <base_sha>..<head_sha>` 的文件数应与元数据 `changed_files` 一致；不一致时以 `--pr <N> --files` 的 API 权威列表为准，禁止据本地 diff 检视。

- `state == "merged"` 时跳过 Step 4，把检视意见整理成 Issue 跟踪（合入后无法按行提交）。

### Step 2 读代码与行号实证

- 一律 `git show <head_sha>:<path>` 读 head 版本代码，**禁止读本地工作区文件**（可能与 head 不一致）。
- 记录每条检视意见前，必须 `git show <head_sha>:<path> | grep -n "<代码片段>"` 取真实行号，**禁止估算行号**。
- findings.json 每条记录 `code_snippet` 字段（该行内容片段），供脚本自动验证行号。

### Step 3 多维检视

**先读检视规范与仓内权威资料，再检视**（检视规范按八类分档，见 [`references/README.md`](./references/README.md) 索引，按 PR 触碰范围加载）：

| 资料 | 用途 | 加载条件 |
|------|------|----------|
| [`references/`](./references/README.md) 检视规范 | 编码安全红线 / 异步与生命周期 / 对外 API / 架构合规 / PR 完备性 | 所有 PR（按触碰范围选载，见索引） |
| [`references/markdown-writing.md`](./references/markdown-writing.md) | Markdown 写作规范（14 类 + 通顺/敏感词/术语） | PR diff 含 `.md` 变更时 |
| [`references/markdown-api-accuracy.md`](./references/markdown-api-accuracy.md) | Markdown 接口文档准确性（原型/参数/样例） | PR 触碰 `docs/zh/api_ref/` 或文档涉及 `include/`、`pkg_inc/` 接口原型时 |
| 根目录 `AGENTS.md` | 架构约束（分层依赖/控制面数据面分离/legacy 约束）、编码规范、目录职责 | 所有 PR |
| `docs/zh/architecture/architecture-brief.md` | 架构权威来源 | PR 触碰 `src/`、`include/`、`pkg_inc/` 时（改前必读） |
| `.clang-format` / `CONTRIBUTING.md` / [CANN 编码规范](https://gitcode.com/cann/community/tree/master/contributor/coding-standards) | 代码风格与贡献规范 | 代码 PR 风格与命名检视时 |
| PR 关联的 Issue 与 PR 描述 | 功能意图（对照实现是否吻合、是否完整） | 所有 PR |

**通用检视维度**（不限于）：正确性（逻辑/边界/未初始化）、资源生命周期（泄漏/Init-Cleanup 对称）、错误处理、并发安全、API 兼容性（include/ 对外接口向后兼容）、安全（越界/注入）、架构合规（对照 AGENTS.md 约束）、命名与风格（对照仓内规范）、注释完整性、性能、测试覆盖（生产代码变更是否补测试）、PR/Issue 描述与实现的吻合度、文档质量（PR 含 .md 变更时对照 markdown-writing / markdown-api-accuracy 规范）、**模块目录文档同步**（新增/变更模块是否有对应的代码级目录文档覆盖，判断口径见 [`references/pr-completeness.md`](./references/pr-completeness.md) 的"模块目录文档同步"节）。

**并行检视**（变更 ≥10 个源文件时）：按模块把文件分组，并行派发多个子 agent，每组返回结构化 JSON findings（schema 见下）。文件少于 10 个时单线程检视。

```json
[{"severity": "CRITICAL|HIGH|MEDIUM|LOW", "file": "src/path.cc", "line": 42,
  "dimension": "Resource Lifecycle", "title": "简短描述",
  "body": "问题描述 + 证据（读到的代码/调用链）+ 影响 + 具体改法（不只点问题）",
  "code_snippet": "该行内容片段"}]
```

### Step 4 汇总与提交

```bash
# 4a 行号验证（不通过会拒绝提交，修正 findings 后重跑）
python3 .agents/skills/hcomm-review/scripts/pr_review.py --pr <N> --verify-only \
  --findings findings.json --repo-root <本仓路径>

# 4b 预演（看每条意见的 position 与回退计划）
python3 .agents/skills/hcomm-review/scripts/pr_review.py --pr <N> --dry-run \
  --findings findings.json --repo-root <本仓路径>

# 4c 提交（自动与已有评论去重；行不在 diff 时自动回退为带 "> file:line" 前缀的 PR 评论）
python3 .agents/skills/hcomm-review/scripts/pr_review.py --pr <N> \
  --findings findings.json --repo-root <本仓路径>

# 4d 检视汇总报告（一条 PR 评论）
python3 .agents/skills/hcomm-review/scripts/pr_review.py --pr <N> --report \
  --findings findings.json --repo-root <本仓路径>
```

提交前对每条意见自问："这是真实问题吗？有无反例？该行实际内容匹配吗？是不是已有代码而非本次变更？"——存疑则不发。

### Step 4e 检视意见处置（修复检视意见后）

修复检视意见（自己提的或他人的）后，**回复必须发到原检视意见的线程内，不要发独立的顶层评论**；自己提的意见修复后要同时 resolve 关闭。一律用脚本完成，不要手工调 API：

```bash
# 列出本账号在该 PR 的检视意见与状态（含 discussion_id）
python3 .agents/skills/hcomm-review/scripts/pr_review.py --pr <N> --list-findings --repo-root <本仓路径>

# 线程回复 + resolve 关闭（自提意见修复后的标准闭环）
python3 .agents/skills/hcomm-review/scripts/pr_review.py --pr <N> --dispose --line <行号>   --body "已修复（commit <sha>）：<修复说明>" --repo-root <本仓路径>

# 只回复不关闭（他人意见、或暂不关闭时）
python3 .agents/skills/hcomm-review/scripts/pr_review.py --pr <N> --dispose --line <行号>   --body "<回复内容>" --no-resolve --repo-root <本仓路径>
```

### Step 5 清理

```bash
python3 .agents/skills/hcomm-review/scripts/pr_review.py --pr <N> --cleanup \
  --worktree <临时目录>/review-<N> --repo-root <本仓路径>
```

移除 worktree、prune、删除临时 findings 文件。

## 检视纪律与意见分级

检视只产出意见、不改被检视代码；所有结论来自实读代码与调用链，不靠关键词命中定罪。

**四条纪律**

- **只读**：不改动被检视代码、不产生任何提交，仓内受版本控制的文件一律不写；只允许写检视中间产物（Step 1 的临时 worktree、findings 等），流程结束按 Step 5 清理。对外产出只有检视意见评论（PR 已合入时为跟踪 Issue）。
- **代码事实**：读完整函数体而非只看 diff，`git show`/grep 核对调用方；按控制流、数据流与调用链判断，不凭推测报告。
- **工具验证**：指针赋值/传递（值传 vs 引用）、算术溢出（确认类型与值域）、结构体成员增删（grep 成员名核对全部引用点）、跨文件遗漏（grep 被改/删符号的全部引用）——这几类发现先工具验证再报。
  - 死代码与注释代码：建议删除未使用的符号/文件前，按全限定名 `git grep` 全仓（含 `test/`、`scripts/`、构建配置）确认无引用；注释掉的代码按「是否有计划保留的理由」判断（注明 Issue 的计划占位不算违规），不做符号引用式的机械判断。对外/ABI 符号、弱符号、dlsym 动态查找符号不得建议删除，确需删除时引导作者走仓内废弃流程（deprecate→宽限期→移除），检视侧只跟踪不代删。
- **反证检验**：报告前反问"什么条件下该问题不成立"，并用工具验证该条件（调用链上游已判空、RAII 已管理该资源、值域受限）；反证成立则不报。

**移动代码与存量代码**

- 与同 PR 删除行内容相同的 `+` 行是移动代码，不按新增违规报；前提是新位置上下文等价（判空、加锁、初始化、生命周期顺序未变），上下文变了并引入新缺陷的仍按本次 diff 定级。
- 检视中读到的存量代码命中任一规则同样提意见，级别按规则本身定（红线类仍是 CRITICAL/HIGH），但不阻塞当次合入；量大可另开 Issue 跟踪。

**严重级别**（每条意见标一个）

| 级别 | 含义 | 与合入的关系 |
|------|------|-------------|
| CRITICAL | 崩溃、数据损坏、安全漏洞、编译/链接失败级缺陷 | 阻塞，必须修复 |
| HIGH | 高风险缺陷，或违反仓内强制规范 | 阻塞，须修复或给出可核实的豁免理由 |
| MEDIUM | 应当修复的质量问题（可读性、可维护性、经验性规范） | 不阻塞，须回应 |
| LOW | 可选改进 | 不阻塞 |

阻塞 = 本次 diff 引入 且 级别为 CRITICAL/HIGH；存量问题无论级别都不阻塞当次合入（见上「移动代码与存量代码」）。表中「与合入的关系」指本次 diff 引入的意见。

**每条意见四要素**：位置（`file:line`，行号实证）、证据（读到的代码或调用链）、影响（会导致什么）、整改建议（具体改法）。缺要素不发。

## 执行纪律

- 全自动执行：不中途询问"是否继续""是否提交"。
- 检视意见**按行单条提交**，每条必须含可操作的修复建议，统一用中文。
- **严禁测试评论**：不以调试/验证为由发任何测试性评论；验证一律用 `--verify-only` / `--dry-run`。
- 提交前脚本自动去重：新意见相互去重，且不与 PR 上已有评论重复。
- **处置检视意见不发独立评论**：回复走原意见线程（`--dispose`），自提意见修复后同时 resolve 关闭。
- **代码变更后同步描述**：PR 内容演进（新增文件/功能/测试）后，同步更新 PR 描述与关联 Issue，保持描述与实现一致（对照 `references/pr-completeness.md` 的吻合度要求）。

## 更多细节

检视规范索引见 [`references/README.md`](./references/README.md)；GitCode API 端点、认证方式、position 语义与已知坑见 [`references/gitcode-api.md`](./references/gitcode-api.md)。

## 评测与回归（维护本 skill 时）

修改本 skill 的规则或元数据后，用 `evals/` 下两份评测集回归验证行为不变（评测不进 CI、不被自动执行，属手动回归基线，由修改规则的人在提交前自行跑）：

- **改 description 触发词后** → 用 `evals/trigger-eval.json`（26 条应/不应触发查询，条数以文件为准）逐条映照，确认触发判断不劣化；与 hcomm-doc-dev 的触发评测集互有镜像对，边界用例两边对照。
- **改"模块目录文档同步"判定规则后**（`references/pr-completeness.md`）→ 按 `evals/evals.json` 前 4 条行为评测走一遍新规则，断言结论与 assertions 一致：新增模块无文档→1 条 LOW 锚主入口；改动使既有陈述失效→MEDIUM；改动落在六段应覆盖但文档缺章→MEDIUM 建议补齐；纯小改动→不提意见。
- **改检视通用流程规则后** → 另按 `evals/evals.json` 后 2 条通用流程用例回归：merged PR 的意见走 Issue 不走行内；新意见与已有检视评论去重后再提。
