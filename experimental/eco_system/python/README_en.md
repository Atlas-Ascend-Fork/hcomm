# hcomm4py — communication-domain Python bindings

[简体中文](./README.md) | English

## Scope (task book §3.1.1, 5 APIs)

| Python | C API |
|---|---|
| `get_root_info` | `HcclGetRootInfo` |
| `comm_config_init` | `HcclCommConfigInit` (header inline; reimplemented in Python) |
| `comm_init_root_info_config` / `init_rank_config` | `HcclCommInitRootInfoConfig` |
| `comm_init_cluster_info_config` | `HcclCommInitClusterInfoConfig` |
| `comm_destroy` | `HcclCommDestroy` |

The 14 collective/P2P ops and function/perf suites live in the sister repo
`cann/hccl` (task issue: https://gitcode.com/cann/hccl/issues/875). Op API
boundaries: sister MR [cann/hccl#3798](https://gitcode.com/cann/hccl/merge_requests/3798)
(after merge: that repo's `experimental/eco_system/python/PUBLIC_API.md`).

## Layout

```text
experimental/eco_system/python/
├── PUBLIC_API.md / PUBLIC_API_en.md  # public API boundary
├── setup.py                    # wheel packaging (package name hcomm4py)
├── hcomm4py/          # domain APIs (only top-level name; avoids clashing with hccl's hccl4py)
├── examples/01_basic/
│   └── 01_comm_init.py
├── tests/
└── README.md
```

Design RFC: [`docs/en/rfcs/0004-hcomm4py-python-bindings.md`](../../../docs/en/rfcs/0004-hcomm4py-python-bindings.md)

## Usage

```bash
source /usr/local/Ascend/cann/set_env.sh   # sets ASCEND_HOME_PATH; CANN >= 9.2.0
cd experimental/eco_system/python
# optional: python3 -m build  # via setup.py && pip install dist/hcomm4py-*.whl
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python3 -c "from hcomm4py import get_root_info, comm_config_init; print('ok', comm_config_init())"
```

### Combining with hccl ops (important)

This repo exports **only** `hcomm4py` (no compat `hccl4py` package) so it cannot
overwrite `cann/hccl`'s ops package on install / `PYTHONPATH`. Use:

```bash
export PYTHONPATH="<hccl/.../python>:<hcomm/.../python>:${PYTHONPATH}"
python3 -c "import hcomm4py as domain; import hccl4py as ops; print(hasattr(ops,'all_reduce'))"
```

`hccl4py` depends on this package: top-level and `hccl4py.types` import domain
APIs/types from `hcomm4py` (no in-tree `hccl4py/comm.py`), sharing one domain
implementation and `libhcomm` load. Acceptance suites and operator examples live
in the hccl repo.

Library search prefers `HCCL4PY_LIB_DIR` / `ASCEND_HOME_PATH` / `LD_LIBRARY_PATH`
(no hardcoded install paths).

## Offline unit test (this repo)

Covers `_OFF` gold offsets, `sizeof` guards, `fill_comm_config` default v11
writing `sqDepth=NOT_SET` (scq stays 0), v12 writing `scqDepth`, and
`logical_size` bounds:

```bash
cd experimental/eco_system/python
PYTHONPATH="$(pwd):${PYTHONPATH}" python3 -m unittest discover -s tests -v
```

## 2-rank domain E2E (this repo)

```bash
source /usr/local/Ascend/cann/set_env.sh
cd experimental/eco_system/python
PYTHONPATH="$(pwd):${PYTHONPATH}" python3 tests/run_domain_e2e.py
```

Covers: `get_root_info` → `comm_config_init` → 2-rank `comm_init_root_info_config`
→ `comm_destroy`.

## Status

- Validated on Ascend 950PR (A5) + CANN 9.2.0-beta.2: RootInfo / ClusterInfo
  create+destroy already covered by hccl-side suites; this repo also ships
  `tests/run_domain_e2e.py`.
- `comm_config_init` layout: explicit size wins version selection over
  capability; capability selects version only when size is omitted. With an
  explicit size, out-of-range capability still emits a version-only
  `UserWarning` (chosen version stays size-derived). When size is omitted and
  capability is out of [10, 12], a `UserWarning` is emitted and version falls
  back to DEFAULT v11. Offline / bind-failure is silent. Implementation:
  `resolve_comm_config_layout` in `hcomm4py/types.py` (not a public API).
