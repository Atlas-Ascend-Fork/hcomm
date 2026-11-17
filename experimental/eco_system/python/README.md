# hcomm4py — HCOMM 通信域 Python 绑定

简体中文 | [English](./README_en.md)

## 本目录覆盖（任务书 §3.1.1，5 个）

| Python | C API |
|---|---|
| `get_root_info` | `HcclGetRootInfo` |
| `comm_config_init` | `HcclCommConfigInit`（头文件 inline，Python 重实现） |
| `comm_init_root_info_config` / `init_rank_config` | `HcclCommInitRootInfoConfig` |
| `comm_init_cluster_info_config` | `HcclCommInitClusterInfoConfig` |
| `comm_destroy` | `HcclCommDestroy` |

通信算子（14 个）与功能/性能套件在姊妹仓 `cann/hccl`（任务 Issue：
https://gitcode.com/cann/hccl/issues/875）。算子 API 边界见
姊妹仓 [cann/hccl#3798](https://gitcode.com/cann/hccl/merge_requests/3798)
（合入后见该仓 `experimental/eco_system/python/PUBLIC_API.md`）。

## 目录

```text
experimental/eco_system/python/
├── PUBLIC_API.md / PUBLIC_API_en.md  # 对外 API 边界（管控）
├── setup.py                    # whl 打包（包名 hcomm4py）
├── hcomm4py/          # 通信域 API（本仓唯一顶层包名，避免与 hccl 的 hccl4py 冲突）
├── examples/01_basic/
│   └── 01_comm_init.py
├── tests/
└── README.md
```

方案文档：[`docs/zh/rfcs/0004-hcomm4py-python-bindings.md`](../../../docs/zh/rfcs/0004-hcomm4py-python-bindings.md)

## 用法

```bash
source /usr/local/Ascend/cann/set_env.sh   # 设置 ASCEND_HOME_PATH 等；CANN >= 9.2.0
cd experimental/eco_system/python
# 可选：python3 -m build  # via setup.py && pip install dist/hcomm4py-*.whl
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python3 -c "from hcomm4py import get_root_info, comm_config_init; print('ok', comm_config_init())"
```

### 与 hccl 算子联用（重要）

本仓**只**导出 `hcomm4py`，不再提供同名 `hccl4py` 兼容包，避免与 `cann/hccl` 的算子包
`hccl4py` 在安装/`PYTHONPATH` 下互相覆盖。正确用法：

```bash
export PYTHONPATH="<hccl/.../python>:<hcomm/.../python>:${PYTHONPATH}"
python3 -c "import hcomm4py as domain; import hccl4py as ops; print(hasattr(ops,'all_reduce'))"
```

`hccl4py` 已改为直接依赖本包：顶层与 `hccl4py.types` 从 `hcomm4py` 导入域 API/类型，同进程共用
同一份域实现与 `libhcomm` 加载（本仓无 `hccl4py/comm.py`）。验收套件在 hccl 仓跑；算子 example 也只放在 hccl 仓。

库搜索优先 `HCCL4PY_LIB_DIR` / `ASCEND_HOME_PATH` / `LD_LIBRARY_PATH`，不写死安装路径。

## 离线单测（本仓）

覆盖 `_OFF` 金标偏移、`sizeof` 守护、`fill_comm_config` 默认 v11 写入 sqDepth=NOT_SET（scq 保持 0）、v12 写入 scqDepth，以及 `logical_size` 边界：

```bash
cd experimental/eco_system/python
PYTHONPATH="$(pwd):${PYTHONPATH}" python3 -m unittest discover -s tests -v
```

## 本仓 2 卡域 E2E

```bash
source /usr/local/Ascend/cann/set_env.sh
cd experimental/eco_system/python
PYTHONPATH="$(pwd):${PYTHONPATH}" python3 tests/run_domain_e2e.py
```

覆盖：`get_root_info` → `comm_config_init` → 2-rank `comm_init_root_info_config` → `comm_destroy`。

## 联调状态

- A5（Ascend 950PR）+ CANN 9.2.0-beta.2 上，RootInfo / ClusterInfo 建域与销毁已经 hccl 侧套件验证通过；本仓亦提供 `tests/run_domain_e2e.py` 自证。
- `comm_config_init` 布局：显式 size 优先于 capability 选版；capability 仅在未给 size 时用于选版；显式 size 下越界 capability 仍发版本告警（所选 version 仍按 size）；未给 size 且 capability 越界则告警并回退 DEFAULT v11；离线静默。实现见 `hcomm4py/types.py` 的 `resolve_comm_config_layout`（非公开 API）。
