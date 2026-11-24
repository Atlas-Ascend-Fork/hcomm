# HcclConfigType

## 功能说明

配置通信算子的展开模式、通信算法配置字符串、UB多channel数量、确定性计算等级、UDI（用户自定义标识）及AI CPU保序任务下发开关。

## 定义原型

```c
typedef enum {
    HCCL_CONFIG_TYPE_INVALID                    = -1,   /* 无效配置项类型 */
    HCCL_CONFIG_TYPE_OP_EXPANSION_MODE          = 0,    /* 算子展开模式，对应类型为hcclOpExpansionMode */
    HCCL_CONFIG_TYPE_HCCL_ALGO                  = 1,    /* 通信算法配置字符串，对应类型为长度HCCL_COMM_ALGO_MAX_LENGTH的char数组, 新增枚举字段向后兼容，不影响旧版本的代码 */
    HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM       = 2,    /* UB多channel数量，对应类型为uint32_t */
    HCCL_CONFIG_TYPE_DETERMINISTIC              = 3,    /* 确定性计算等级，对应类型为uint32_t */
    HCCL_CONFIG_TYPE_UDI                        = 4,    /* 用户自定义标识（UDI），对应类型为长度UDI_MAX_LENGTH的char数组，当前仅Atlas训练系列产品支持查询 */
    HCCL_CONFIG_TYPE_AICPU_ORDER_LAUNCH_ENABLE  = 5,    /* AI CPU保序任务下发开关，取自环境变量HCCL_AICPU_ORDER_LAUNCH_ENABLE，对应类型为uint32_t（1=开启，0=关闭），默认开启 */
} HcclConfigType;
```
