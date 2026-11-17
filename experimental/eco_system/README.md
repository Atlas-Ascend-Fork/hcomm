# eco_system

简体中文 | [English](./README_en.md)

本目录存放生态实验代码。`python/` 下是 **hcomm4py**（HCOMM 通信域 Python 绑定，
任务书 §3.1.1，5 个 API）。本仓不提供顶层 `hccl4py` 包名。

通信算子绑定与功能/性能验收套件在姊妹仓 `cann/hccl`
（Issue：https://gitcode.com/cann/hccl/issues/875）。

## 快速入口

```bash
cd python
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python3 examples/01_basic/01_comm_init.py --nranks 2
# 算子 example 仅在 cann/hccl 仓维护
```
