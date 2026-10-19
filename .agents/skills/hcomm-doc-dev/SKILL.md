---
name: hcomm-doc-dev
description: 本 skill 是 hcomm 仓文档开发域的总入口与路由器：按文档类型将请求路由到对应的子规范。当用户要求"写模块文档""解析某目录代码""给 XX 目录出代码级说明""补 README""出中英文对照的模块文档"，或咨询"某目录改动后模块文档要不要更新/补齐"时，应使用本 skill——此类软件模块/组件目录的代码级解析文档请求路由到 references/software-module/ 子规范；未来补充其他文档类型（如接口 API 文档）时同样以 references/ 下子目录形式挂接。docs/ 目录下的文档写作、PR 检视中的意见判定不适用（后者归 hcomm-review）。
---

# Hcomm Doc Dev（文档开发路由）

## Overview

hcomm 仓的**文档开发域总入口**：本 SKILL.md 只做一件事——判断请求属于哪类文档开发，路由到对应子规范；子规范承载该类型的完整写作方法论（骨架、流程、校验、模板）。**不要在本文件中寻找写作规则，路由表能回答你的第一个问题，其余一切在子规范里。**

## 路由表

| 文档类型 | 特征 | 路由目标 |
|---------|------|---------|
| **软件模块/组件代码级解析文档** | 输入是源码目录（如 hcomm 的 dfx/taskException、dfx/profiling、communicator），产出随代码目录归档的六段骨架中英双语文档（模块名.md + 模块_en.md）；含"存量模块文档要不要更新"的咨询 | `references/software-module/` —— 主规范 `module-doc.md`（探查/结构/写作/双语/三层校验五步流程） |
| （暂无）接口 API 文档 | docs/ 下的对外接口参考类文档 | 暂无子规范。触发时**如实告知**："接口 API 文档类型暂无开发规范"，不套用 software-module 六段骨架 |
| （暂无）用户指南/构建/教程 | docs/ 下的指南、教程类文档 | 同上，暂无子规范，如实告知 |

**路由判定原则**：看产出物的落点与形态，不看措辞。用户说"写文档"但目标是 docs/ 下的指南 → 告知暂无规范；用户说"补 README"但目标在模块源码目录（如 cluster_monitor/）→ 路由 software-module。

## 路由后动作

命中 software-module 后：
1. 读 `references/software-module/module-doc.md`（完整写作流程）；
2. 按其指引按需加载同目录 writing-rules.md / mermaid-guide.md / terminology.md 与 assets/、scripts/、evals/ 对应资源——均为 `software-module` 子目录下的同名文件。

## 扩展新文档类型（维护指引）

新增一类文档开发能力时（如接口 API 文档）：在 references/、assets/、scripts/、evals/ 下各建同名子目录（如 api-doc/），子规范主文件命名 `<类型>-doc.md`，在本路由表加一行；本 SKILL.md 只加路由行，不沉淀任何写作规则——规则一律进子规范。

## 触发评测（维护用）

- `evals/trigger-eval.json` — 域级触发评测集（27 条应/不应触发查询；该请求是否属于文档开发域、应唤醒本 skill。仅维护用：改 description 触发词后逐条映照回归，写作流程中不使用；条数以文件为准）
- `evals/route-eval.json` — 路由评测集（10 条；已触发后应路由到哪个子规范。仅维护用：改路由表后逐条映照回归）
- `evals/software-module/evals.json` — software-module 类型行为评测集（5 条：六段骨架/代号规避/存量改名/多组件暂停/漂移行号修正；仅维护用：改写作规则后按场景走一遍断言一致；条数以文件为准）
