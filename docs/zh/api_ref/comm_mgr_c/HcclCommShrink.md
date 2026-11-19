# HcclCommShrink

> [!NOTE]说明
> 本接口为试验特性，使用`HCCL_EXPERIMENTAL`宏标识，后续版本可能在不另行通知的情况下调整、重命名或移除，不承诺向后兼容。默认不产生编译期告警；如需开启试验特性告警，可定义`HCCL_ENABLE_EXPERIMENTAL_WARNING`。开启告警后，若编译选项同时开启`-Werror`（或MSVC `/WX`），需要在调用点局部关闭告警。

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

从已有通信域中剔除指定rank，由存活rank创建一个规模更小的新通信域。该接口不会原地修改传入的通信域，而是通过`newComm`返回新的通信域句柄。

缩容后的rank编号按照原通信域rank ID升序重新连续编号。例如原通信域包含rank 0~7，剔除rank 2和rank 5后，原rank 3在新通信域中的编号为2。

## 函数原型

```c
HcclResult HcclCommShrink(
    HcclComm comm,
    const uint32_t *excludeRanksList,
    const uint32_t excludeRanksCount,
    const HcclCommConfig *config,
    uint32_t shrinkFlags,
    HcclComm *newComm)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 待缩容的通信域句柄。该通信域必须由rootInfo方式创建，且为当前最新一代通信域。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| excludeRanksList | 输入 | 待剔除的rank ID数组。rank ID为当前通信域中的编号。<br>数组不能为空，不能包含重复rank ID或调用者自身。所有存活rank必须传入相同内容，建议按rank ID升序排列。 |
| excludeRanksCount | 输入 | 待剔除的rank数量。取值范围为(0, 当前通信域rank数量)。 |
| config | 输入 | 新通信域配置。<br>传`NULL`时继承父通信域配置；非`NULL`时，建议先调用[HcclCommConfigInit](HcclCommConfigInit.md)初始化，未显式设置的字段使用默认值。所有存活rank必须传入相同配置。<br>HcclCommConfig类型的定义可参见[HcclCommConfig](./data_type_definition/HcclCommConfig.md)。 |
| shrinkFlags | 输入 | 缩容标志，当前为保留参数，必须传入0。 |
| newComm | 输出 | 缩容后的新通信域句柄。接口成功时有效。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

| 返回值 | 说明 |
| --- | --- |
| HCCL_SUCCESS | 缩容成功，`newComm`返回新通信域句柄。 |
| HCCL_E_PTR | `comm`、`excludeRanksList`或`newComm`为空指针，或通信域内部对象异常。 |
| HCCL_E_PARA | 剔除列表为空、剔除数量非法、rank ID越界、剔除列表中包含调用者自身、配置非法或派生通信域名称超长等。 |
| HCCL_E_NOT_SUPPORT | 通信域由rankTable方式创建，或传入的通信域不是当前最新一代通信域。 |
| HCCL_E_INTERNAL | 创建新通信域或内部资源初始化失败。 |

## 约束说明

- 仅支持Ascend 950PR&950DT系列产品。
- 仅支持由rootInfo方式创建的通信域。由[HcclCommInitClusterInfo](HcclCommInitClusterInfo.md)或[HcclCommInitClusterInfoConfig](HcclCommInitClusterInfoConfig.md)创建的通信域不支持缩容。
- 缩容前必须同步所有相关Stream，确保`comm`上没有在途集合通信、点对点通信或资源操作。
- 所有存活rank必须调用本接口，并传入相同的`excludeRanksList`、`excludeRanksCount`和`config`。
- 所有参与者必须在同一轮缩容中完成调用。任一存活rank未调用或参数不一致，都会导致各rank的通信域成员视图不一致。
- 被剔除的rank不得调用本接口。
- 每次缩容必须基于当前最新一代通信域。缩容成功后，旧通信域不再支持继续缩容或扩容。
- 如果缩容剔除了原通信域的root，新通信域在缩容成功后没有有效root。后续扩容前，需要由一个存活rank调用[HcclCommGetRootInfo](HcclCommGetRootInfo.md)触发root重选，并将新的`rootInfo`分发给所有扩容参与者。
- 本接口不支持多线程并发调用，调用方须保证不同线程对本接口的调用串行执行。
- 默认不产生编译期告警。若定义`HCCL_ENABLE_EXPERIMENTAL_WARNING`开启试验特性告警，且编译选项同时开启`-Werror`（或MSVC `/WX`），告警会导致编译失败。请仅在调用点附近局部抑制告警，并在调用后恢复告警设置。例如，GCC/Clang可使用`#pragma GCC diagnostic ignored "-Wattribute-warning"`，MSVC可使用`#pragma warning(disable: 4996)`。
- 本接口不会自动销毁旧通信域。业务侧应在确认旧通信域不再使用后调用[HcclCommDestroy](HcclCommDestroy.md)。
- 缩容后，对称内存窗口、CCU实例、Channel、Thread等资源不会自动迁移到新通信域，需在新通信域上重新注册或获取。
- `comm`必须为有效的通信域句柄，禁止传入空指针、野指针或已销毁的句柄。


## 调用示例

```c
// 试验特性：本接口后续版本可能调整或移除，不承诺向后兼容。
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// 待缩容通信域和任务流，由业务按实际场景设置
HcclComm comm = NULL;
aclrtStream stream = NULL;
HcclComm newComm = NULL;

// 确保旧通信域上没有在途任务
ACLCHECK(aclrtSynchronizeStream(stream));

// 剔除原通信域中的rank 2和rank 5，所有存活rank必须传入相同列表
uint32_t excludeRanksList[] = {2, 5};
const uint32_t excludeRanksCount = 2;

// config传NULL时继承父通信域配置
HCCLCHECK(HcclCommShrink(comm, excludeRanksList, excludeRanksCount, NULL, 0, &newComm));

// 缩容后使用新通信域查询rank信息
uint32_t rankId = 0;
uint32_t rankSize = 0;
HCCLCHECK(HcclGetRankId(newComm, &rankId));
HCCLCHECK(HcclGetRankSize(newComm, &rankSize));
```
