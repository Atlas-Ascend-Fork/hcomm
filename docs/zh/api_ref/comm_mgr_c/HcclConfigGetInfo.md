# HcclConfigGetInfo

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas训练系列产品：不支持
<!-- end id5 -->

## 功能说明

获取指定通信域的HCCL配置信息。

根据配置项类型查询对应的配置信息，并写入调用者提供的缓冲区中，当前支持查询通信算子的展开模式、通信算法配置字符串、UB多channel数量、确定性计算等级、UDI及AI CPU保序任务下发开关。

## 函数原型

```c
HcclResult HcclConfigGetInfo(HcclComm comm, HcclConfigType cfgType, uint32_t infoLen, void *info);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| cfgType | 输入 | 需要查询的配置项类型，HcclConfigType的定义可参见[HcclConfigType](./data_type_definition/HcclConfigType.md)。 |
| infoLen | 输入 | 目标配置类型的大小（字节数）。查询HCCL_CONFIG_TYPE_OP_EXPANSION_MODE时必须等于待查询配置类型的实际大小；查询HCCL_CONFIG_TYPE_HCCL_ALGO时必须不小于HCCL_COMM_ALGO_MAX_LENGTH；查询HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM、HCCL_CONFIG_TYPE_DETERMINISTIC或HCCL_CONFIG_TYPE_AICPU_ORDER_LAUNCH_ENABLE时必须等于sizeof(uint32_t)；查询HCCL_CONFIG_TYPE_UDI时必须不小于UDI_MAX_LENGTH字节数（该配置项当前仅Atlas训练系列产品支持查询）。 |
| info | 输出 | 配置信息输出缓冲区，必须按目标配置类型对齐且可写。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。
- 查询HCCL_CONFIG_TYPE_DETERMINISTIC时，返回指定通信域的有效配置：0或1表示普通确定性计算，2表示严格确定性计算，即保序功能。
- 查询HCCL_CONFIG_TYPE_AICPU_ORDER_LAUNCH_ENABLE时，返回AI CPU保序任务下发开关（1=开启，0=关闭），取自进程级环境变量HCCL_AICPU_ORDER_LAUNCH_ENABLE，取值范围为1或0，默认值为1（开启）。该环境变量在通信域初始化时解析，配置非法会导致通信域初始化失败。

## 调用示例

```c
HcclConfigTypeOpExpansionMode mode;
uint32_t size = sizeof(HcclConfigTypeOpExpansionMode); // 必须等于目标类型大小
HcclResult ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, size, &mode);

// 查询通信算法字符串
char algoInfo[HCCL_COMM_ALGO_MAX_LENGTH];
uint32_t algoSize = HCCL_COMM_ALGO_MAX_LENGTH; // 必须不小于HCCL_COMM_ALGO_MAX_LENGTH
ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_HCCL_ALGO, algoSize, algoInfo);

// 查询UB多channel数量
uint32_t multiChannelNum = 0;
uint32_t numSize = sizeof(uint32_t); // 必须等于目标类型大小
ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM, numSize, &multiChannelNum);

// 查询确定性计算等级
uint32_t deterministic = 0;
uint32_t deterministicSize = sizeof(uint32_t); // 必须等于目标类型大小
ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_DETERMINISTIC, deterministicSize, &deterministic);

// 查询UDI
char udiInfo[UDI_MAX_LENGTH];
uint32_t udiSize = UDI_MAX_LENGTH; // 必须不小于UDI_MAX_LENGTH
ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_UDI, udiSize, udiInfo);

// 查询AI CPU保序任务下发开关
uint32_t aicpuOrderLaunchEnable = 0;
uint32_t enableSize = sizeof(uint32_t); // 必须等于目标类型大小
ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_AICPU_ORDER_LAUNCH_ENABLE, enableSize, &aicpuOrderLaunchEnable);
```
