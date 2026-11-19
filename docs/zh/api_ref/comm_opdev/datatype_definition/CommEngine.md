# CommEngine

## 功能说明

通信引擎类型枚举。

## 定义原型

```c
typedef enum {
    COMM_ENGINE_RESERVED = -1,    //< 保留的通信引擎
    COMM_ENGINE_CPU = 0,          //< HOST CPU引擎
    COMM_ENGINE_CPU_TS = 1,       //< HOST CPU TS引擎
    COMM_ENGINE_AICPU = 2,        //< AI CPU引擎
    COMM_ENGINE_AICPU_TS = 3,     //< AI CPU TS引擎
    COMM_ENGINE_AIV = 4,          //< AIV引擎
    COMM_ENGINE_CCU = 5,          //< CCU引擎
} CommEngine;
```

## 产品支持情况

<!-- npu="950" id1 -->
针对Ascend 950PR&950DT系列产品，支持的通信引擎如下：

  - COMM_ENGINE_CPU
  - COMM_ENGINE_CPU_TS
  - COMM_ENGINE_AICPU
  - COMM_ENGINE_AICPU_TS
  - COMM_ENGINE_AIV
  - COMM_ENGINE_CCU
<!-- end id1 -->
    
<!-- npu="A3" id2 -->
针对Atlas A3系列产品，支持的通信引擎如下：

  - COMM_ENGINE_AICPU_TS
<!-- end id2 -->

<!-- npu="910b" id3 -->
针对Atlas A2系列产品，支持的通信引擎如下：

  - COMM_ENGINE_CPU_TS
<!-- end id3 -->
