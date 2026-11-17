# eco_system

[简体中文](./README.md) | English

Experimental ecosystem code. The `python/` subdirectory holds **hcomm4py**
(HCOMM communication-domain Python bindings, task book §3.1.1, 5 APIs).
This repo does not ship a top-level `hccl4py` package.

Operator bindings and acceptance suites live in the sister repository `cann/hccl`
(issue: https://gitcode.com/cann/hccl/issues/875).

## Quick start

```bash
cd python
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python3 examples/01_basic/01_comm_init.py --nranks 2
# Operator examples are maintained only in cann/hccl
```
