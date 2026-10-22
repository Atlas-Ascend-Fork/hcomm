# HcclConfig

## 功能说明

定义集合通信相关配置。

## 定义原型

```c
typedef enum {
    HCCL_DETERMINISTIC = 0, /* 0: non-deterministic, 1: deterministic, 2: strict(order-preserving) */
    HCCL_CONFIG_RESERVED
} HcclConfig;
```

## 参数说明

- HCCL_DETERMINISTIC：是否开启确定性计算。

  - 0：不开启确定性计算。针对Ascend 950PR&950DT系列产品，归约类通信算子仍使用确定性计算。
  - 1：开启确定性计算。
  - 2：开启保序功能，支持Ascend 950PR&950DT系列产品、Atlas A3系列产品和Atlas A2系列产品。各产品支持的算子和使用约束请参见[HCCL_DETERMINISTIC](https://gitcode.com/cann/hccl/blob/9.3.0-beta.1/docs/zh/user_guide/hccl_env/HCCL_DETERMINISTIC.md)。

- HCCL_CONFIG_RESERVED：预留参数。
