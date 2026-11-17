# RFC：hcomm4py — HCOMM 通信域 Python 绑定

- 起始日期：2026-10-14
- RFC PR编号：[cann/hcomm#6331](https://gitcode.com/cann/hcomm/merge_requests/6331)
- 相关Issue：社区任务历史上称「实现 hccl4py」，本 MR 实际交付包名为 `hcomm4py`；讨论帖 https://gitcode.com/cann/hccl/discussions/19

---

## 概要

在 `experimental/eco_system/python` 以 ctypes 封装 HCOMM 侧 5 个通信域 C API，形成 Python 包 `hcomm4py`（本仓唯一顶层包名，不再提供 `hccl4py` 兼容包，避免与 cann/hccl 的算子包冲突）。对外边界见 `PUBLIC_API.md`，可通过 `setup.py` 构建 whl。算子层见姊妹仓 hccl RFC `0004-hccl4py-python-bindings`。

## 背景与动机

- 通信域创建/配置/销毁是集合通信的前置步骤，需要与算子绑定同样可维护的 Python 面。
- 双仓同名 `hccl4py` 无法靠 `PYTHONPATH` 合并；本仓以 `hcomm4py` 为权威域名，避免与 hccl 算子包冲突。

## 术语表

| 术语 | 含义 |
|------|------|
| hcomm4py | 本仓唯一顶层包名（通信域） |
| hccl4py（hccl 仓） | 姊妹仓 cann/hccl 的算子包；联用时与本仓 `hcomm4py` 搭配（本仓不提供同名兼容包） |

## 架构与接口契约

### 整体架构

```text
experimental/eco_system/python/
├── PUBLIC_API.md / PUBLIC_API_en.md
├── setup.py
├── hcomm4py/          # 本仓唯一顶层包
│   ├── __init__.py
│   ├── _lib.py
│   ├── types.py
│   ├── _validate.py
│   └── comm.py
├── examples/
└── tests/
```

### 对外接口

| Python | C API |
|---|---|
| `get_root_info` | `HcclGetRootInfo` |
| `comm_config_init` | `HcclCommConfigInit`（Python 重实现） |
| `comm_init_root_info_config` / `init_rank_config` | `HcclCommInitRootInfoConfig` |
| `comm_init_cluster_info_config` | `HcclCommInitClusterInfoConfig` |
| `comm_destroy` / `Comm` | `HcclCommDestroy` + 上下文管理器 |

入参格式校验：`nranks`/`rank`/`root_info`/`cluster_info`/`config` 在进入 native 前检查。

### 依赖接口

`libhcomm.so`（及可选 `libascendcl.so`），搜索顺序见 `_lib.py`。

## 影响分析

- 纯新增 `experimental/`，不影响既有 C API。
- 可选 whl 发布；CMake 主产物不变。
- 与 hccl 算子联用需避免双 CDLL 混用句柄。

## 兼容性考虑

- 本仓仅发布 `hcomm4py`；与 hccl 仓 `hccl4py` 算子包通过 `PYTHONPATH` / 分别安装共存，不提供本仓同名兼容包。
- `comm_config_init` 布局由 `resolve_comm_config_layout` 决定：**显式 `version`/`logical_size` > 运行时 `HcclGetCommConfigCapability()`（返回值∈[10,12] 时作为 config version）> 默认 v11/2248**。有显式 `logical_size` 且未显式 `version` 时，按该 size 选可覆盖的最高 version（如 2240–2243 → v10、2244–2247 → v11、2248 → v12），**优先于** capability（含 cap∈[10,12]）。capability **仅在未给 `logical_size` 时用于选版**；未给 size 且 capability 越界时发 `UserWarning` 并回退 DEFAULT v11；显式 size 下若 capability 越界仍发版本-only `UserWarning`，所选 version 仍按 size 推导（不回落 DEFAULT v11）。库不可用/绑定失败（`capability=None`）按 size 静默选版。显式 `version` 越界仍抛 `ValueError`。**不是**写死的「CANN 发行版→version」对照表。`logical_size` 缺省时再按 version 映射（v10→2240，v11/v12→2248）。权威细节以 `resolve_comm_config_layout` docstring 为准。

## 详细设计

- `fill_comm_config` + `_OFF` 偏移表维护 ABI；版本门限由偏移推导；字段覆盖走 `require_u32` / `require_int` 值域校验。
- `Comm.destroy()` 幂等；`with Comm` 异常路径仍销毁句柄。
- 离线 UT 接入本仓 `.gitcode/scripts/ut.sh`；可选 `offsetof` 探针在可编译 `hccl_types.h` 时与 `_OFF` 金表比对。

### 打包

```bash
cd experimental/eco_system/python
python3 -m build  # via setup.py
pip install dist/hcomm4py-*.whl
```

## 测试方案

| 项 | 方法 |
|----|------|
| 离线 | `tests/test_comm_*_offline.py`（含 Config ABI 金表与可选头文件 offsetof 探针） |
| 域 E2E | `tests/run_domain_e2e.py`（≥2 device，ACL 返回值显式检查） |
| 示例 | `examples/01_basic/01_comm_init.py` 等 |
| 联用 | 与 hccl 仓 `hccl4py` 算子同机验收 |

## 风险评估

| 风险 | 应对 |
|------|------|
| 包名冲突 | 权威名 `hcomm4py` + 文档约束 |
| Config ABI 漂移 | `_OFF` + 版本门控 + 离线 UT |

## 替代方案

ctypes（采用） / pybind11 / 仅文档。

## 开放问题

1. ~~是否在发行面同时发布 `hcomm4py` 与 `hccl4py` 两个 dist 名。~~ **已决策：否**——本仓只发布 `hcomm4py`，`hccl4py` 仅由 cann/hccl 承载算子包。
2. 与 hccl 仓算子包的长期收敛（hccl 侧已无 in-tree 域实现，仅重导出本仓 API）。

---

## 评审记录

详见 cann/hcomm#6331。
