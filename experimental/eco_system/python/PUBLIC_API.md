# hcomm4py 对外 API 边界

简体中文 | [English](./PUBLIC_API_en.md)

本文件是 **hcomm4py 对外边界清单**，供合入后管控与评审对照。  
**仅 `hcomm4py/__init__.py` 的 `__all__` 中符号为稳定对外 API**；下划线前缀模块/符号为内部实现。本仓不提供 `hccl4py` 兼容包名（避免与 cann/hccl 算子包冲突）。

## 边界文件映射

| 职责 | 对应文件 | 说明 |
|---|---|---|
| 对外导出入口 | `hcomm4py/__init__.py` | 唯一稳定导入面（`__all__`） |
| 类型 / 枚举 / 配置布局 | `hcomm4py/types.py` | 与 C 头布局对齐 |
| 通信域 API | `hcomm4py/comm.py` | 任务书 §3.1.1，5 个域接口 |
| 动态库加载 | `hcomm4py/_lib.py` | **内部**，非对外 API |
| 打包入口 | `setup.py` | whl 构建入口（`python -m build` / `pip wheel .`） |

## 对外符号 ↔ C API

### 类型

| Python | C / 头文件概念 |
|---|---|
| `HcclResult` / `HcclError` | `HcclResult` |
| `HcclDataType` | `HcclDataType` |
| `HcclReduceOp` | `HcclReduceOp` |
| `HcclSendRecvType` | `HcclSendRecvType` |
| `HcclRootInfo` | `HcclRootInfo` |
| `HcclCommConfig` | `HcclCommConfig` |
| `HcclSendRecvItem` | `HcclSendRecvItem` |
| `HCCL_ROOT_INFO_BYTES` | `HCCL_ROOT_INFO_BYTES`（4108） |
| `HCCL_COMM_CONFIG_BYTES` | Python raw buffer 容量（2304） |
| `HCCL_COMM_CONFIG_MAGIC_WORD` | Config magic `0xF0F0F0F0` |
| `HCCL_COMM_CONFIG_SIZE_V10` / `V11` / `V12` | 逻辑 size 门限（2240 / 2248 / 2248） |
| `HCCL_COMM_CONFIG_VERSION_DEFAULT` | 离线回退 version（11） |
| `HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT` | 离线回退 logical_size（= V11） |

### 通信域

通信域 API 经 ctypes 从运行时 `libhcomm.so` 解析非 V2 符号：`HcclGetRootInfo`、`HcclCommInitRootInfoConfig`、`HcclCommInitClusterInfoConfig`、`HcclCommDestroy`。  
这些导出对应 legacy A910 / op_base 路径；Ascend 950PR / op_base_v2 若仅提供 `*V2` 符号则与本绑定不兼容（属平台/导出差异，不是「升级到更新的 CANN 发行版，例如超越 9.1.0」可解决的版本问题）。符号缺失时 `_bind()` 抛带平台说明的 `RuntimeError`。

| Python | C API |
|---|---|
| `get_root_info` | `HcclGetRootInfo` |
| `get_comm_config_capability` | `HcclGetCommConfigCapability`（可选；不可用时返回 `None`） |
| `comm_config_init` | `HcclCommConfigInit`（头文件 inline，Python 重实现） |
| `comm_init_root_info_config` / `init_rank_config` | `HcclCommInitRootInfoConfig` |
| `comm_init_cluster_info_config` | `HcclCommInitClusterInfoConfig` |
| `comm_destroy` | `HcclCommDestroy` |
| `Comm` | 句柄上下文管理器（Python 侧） |

集合/点对点算子边界见姊妹仓 cann/hccl：
姊妹仓 [cann/hccl#3798](https://gitcode.com/cann/hccl/merge_requests/3798)
（合入后见该仓 `experimental/eco_system/python/PUBLIC_API.md`）。

## 变更管控约定

1. 新增/删除/重命名对外符号：必须同步更新本文件与 `__all__`。
2. 仅内部重构：可不改本文件，但不得泄漏到 `__all__`。
3. 与 hccl 算子联用：`import hcomm4py` 建域 / `import hccl4py`（hccl 仓）调算子。
