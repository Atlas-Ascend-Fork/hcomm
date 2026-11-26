# RFC: hcomm4py — HCOMM Domain Python Bindings

- Start date: 2026-10-14
- RFC PR: [cann/hcomm#6331](https://gitcode.com/cann/hcomm/merge_requests/6331)
- Related: community task historically titled “implement hccl4py”; this MR delivers package `hcomm4py`; discussion https://gitcode.com/cann/hccl/discussions/19

---

## Summary

Provide ctypes bindings for 5 HCOMM domain C APIs as package `hcomm4py` (sole top-level package in this repo; no compat `hccl4py` package, to avoid colliding with cann/hccl ops). Public surface is governed by `PUBLIC_API.md`; wheels via `setup.py`. Ops live in sister hccl RFC `0004-hccl4py-python-bindings`.

## Motivation

Domain create/config/destroy is required before collectives. Dual-repo top-level name `hccl4py` cannot be merged via `PYTHONPATH`; this repo uses canonical `hcomm4py`.

## Terminology

| Term | Meaning |
|------|---------|
| hcomm4py | Sole top-level domain package in this repo |
| hccl4py (hccl repo) | Ops package in sister cann/hccl; pair with `hcomm4py` when combining (this repo does not ship a compat package) |

## Architecture and Contracts

### Layout

```text
experimental/eco_system/python/
├── PUBLIC_API.md / PUBLIC_API_en.md
├── setup.py
├── hcomm4py/    # sole top-level package
│   ├── __init__.py
│   ├── _lib.py
│   ├── types.py
│   ├── _validate.py
│   └── comm.py
├── examples/
└── tests/
```

### Public API

`get_root_info`, `comm_config_init`, `comm_init_root_info_config` / `init_rank_config`, `comm_init_cluster_info_config`, `comm_destroy`, `Comm`, plus shared types/enums. Argument validation runs before native calls.

### Dependencies

`libhcomm.so` (and optional `libascendcl.so`) via env search paths in `_lib.py`.

## Impact / Compatibility

Additive under `experimental/`. This repo publishes only `hcomm4py`; coexist with the hccl-repo `hccl4py` ops package via `PYTHONPATH` / separate installs. `comm_config_init` layout follows `resolve_comm_config_layout`: **explicit `version`/`logical_size` > runtime `HcclGetCommConfigCapability()` (used as the config version when in [10, 12]) > DEFAULT v11/2248**. With an explicit `logical_size` and no explicit `version`, the highest version that size can cover wins over capability — including when capability is in [10, 12] (e.g. 2240–2243 → v10, 2244–2247 → v11, 2248 → v12). Capability is used to *select* version only when `logical_size` is omitted; when `logical_size` is omitted and capability is out of [10, 12], a `UserWarning` is emitted and version falls back to DEFAULT v11. With an explicit `logical_size`, an out-of-range capability still emits a version-only `UserWarning`; the chosen version stays size-derived (e.g. 2240 → v10) and does not fall back to DEFAULT v11. Offline / bind-failure (`capability=None`) picks by size silently. Explicit `version` outside [10, 12] still raises `ValueError`. This is **not** a hardcoded CANN-release→version table. Omitted `logical_size` then follows version (v10→2240, v11/v12→2248). Authoritative detail: `resolve_comm_config_layout` docstring.

## Detailed Design

- `fill_comm_config` + `_OFF` offset table; version floors derived from offsets; field overrides go through `require_u32` / `require_int`.
- Idempotent `Comm.destroy()`; context manager destroys on exception paths.
- Offline UTs wired into `.gitcode/scripts/ut.sh`, including an optional `offsetof` probe against `hccl_types.h` when a C++ toolchain is present.

### Packaging

```bash
cd experimental/eco_system/python
python3 -m build  # via setup.py
pip install dist/hcomm4py-*.whl
```

## Testing

Offline `tests/test_comm_*_offline.py` (ABI gold + optional header offsetof probe), domain E2E `tests/run_domain_e2e.py`, examples, and combined acceptance with hccl ops.

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Package name conflict | Authoritative name `hcomm4py` + documentation constraints |
| Config ABI drift | `_OFF` offset table + version gating + offline UTs |

## Alternatives

ctypes (chosen) / pybind11 / docs-only.

## Open Questions

1. ~~Whether to publish both dist names.~~ **Decided: no** — this repo ships only `hcomm4py`; `hccl4py` remains the ops package in cann/hccl.
2. Long-term convergence with the hccl ops package (hccl no longer ships an in-tree domain; it only re-exports this package).

---

## Review Notes

See cann/hcomm#6331.
