# HcclCommGrow

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

在已有通信域基础上加入新rank，创建一个规模更大的新通信域。该接口不会原地修改传入的通信域，而是通过`newComm`返回新的通信域句柄。

扩容后，原通信域中的存活rank保持原rank编号，新加入rank从原通信域rank数量开始连续追加编号。

## 函数原型

```c
HcclResult HcclCommGrow(
    HcclComm comm,
    uint32_t nRanks,
    const HcclRootInfo *rootInfo,
    uint32_t rank,
    const HcclCommConfig *config,
    HcclComm *newComm)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 原通信域中存活rank传入当前通信域句柄，包括原root被缩容剔除的场景；新加入rank传入`NULL`。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| nRanks | 输入 | 扩容后的通信域总rank数量，必须大于扩容前通信域rank数量。所有参与者必须传入相同值。 |
| rootInfo | 输入 | 扩容后通信域的root信息，由[HcclCommGetRootInfo](HcclCommGetRootInfo.md)获取。所有参与者必须传入相同的`rootInfo`。 |
| rank | 输入 | 本rank在扩容后通信域中的编号。存活rank保持原编号（0~N-1），新rank追加编号（N~N+M-1）。<br>其中N为扩容前rank数量，M为新增rank数量，且`rank < nRanks`。 |
| config | 输入 | 新通信域配置。<br>传`NULL`时使用默认配置；非`NULL`时，建议先调用[HcclCommConfigInit](HcclCommConfigInit.md)初始化，未显式设置的字段使用默认值。所有参与者必须传入相同配置。<br>HcclCommConfig类型的定义可参见[HcclCommConfig](./data_type_definition/HcclCommConfig.md)。 |
| newComm | 输出 | 扩容后的新通信域句柄。接口成功时有效。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

| 返回值 | 说明 |
| --- | --- |
| HCCL_SUCCESS | 扩容成功，`newComm`返回新通信域句柄。 |
| HCCL_E_PTR | `newComm`为空指针，或传入的通信域内部对象异常。 |
| HCCL_E_PARA | `rootInfo`为空、`nRanks`非法、`rank`越界、配置非法或派生通信域名称超长等。 |
| HCCL_E_NOT_SUPPORT | 通信域由rankTable方式创建，或传入的通信域不是当前最新一代通信域。 |
| HCCL_E_INTERNAL | 拓扑信息收集不完整、建链失败或内部资源初始化失败。 |

## 约束说明

- 仅支持Ascend 950PR&950DT系列产品。
- 仅支持基于rootInfo方式创建的通信域。由[HcclCommInitClusterInfo](HcclCommInitClusterInfo.md)或[HcclCommInitClusterInfoConfig](HcclCommInitClusterInfoConfig.md)创建的通信域不支持扩容。
- 扩容前必须同步所有相关Stream，确保原通信域上没有在途集合通信、点对点通信或资源操作。
- 所有参与者必须调用本接口，并传入相同的`nRanks`、`rootInfo`和`config`。
- 所有存活rank和新rank必须在同一轮扩容中完成调用。任一参与者未启动、未调用或参数不一致，可能导致拓扑信息收集不完整、建链失败或等待超时。
- 原通信域中的存活rank传入当前通信域句柄，新加入rank传入`NULL`。
- 扩容前，由业务侧选择一个存活rank调用[HcclCommGetRootInfo](HcclCommGetRootInfo.md)获取`rootInfo`，并分发给所有参与者。不能由多个rank并发触发root重选。
- 如果原通信域的root在缩容时已被剔除，[HcclCommGetRootInfo](HcclCommGetRootInfo.md)会触发root重选并创建新的root server。该操作应由一个存活rank执行，并将结果分发给所有存活rank和新rank。
- 每次扩容必须基于当前最新一代通信域。扩容成功后，旧通信域不再支持继续扩容或缩容。
- 本接口不支持多线程并发调用，调用方须保证不同线程对本接口的调用串行执行。
- 默认不产生编译期告警。若定义`HCCL_ENABLE_EXPERIMENTAL_WARNING`开启试验特性告警，且编译选项同时开启`-Werror`（或MSVC `/WX`），告警会导致编译失败。请仅在调用点附近局部抑制告警，并在调用后恢复告警设置。例如，GCC/Clang可使用`#pragma GCC diagnostic ignored "-Wattribute-warning"`，MSVC可使用`#pragma warning(disable: 4996)`。
- 本接口不会自动销毁旧通信域。业务侧应在确认旧通信域不再使用后调用[HcclCommDestroy](HcclCommDestroy.md)。
- 扩容后，对称内存窗口、CCU实例、Channel、Thread等资源不会自动迁移到新通信域，需在新通信域上重新注册、绑定或获取。
- 传入的`comm`必须为有效的通信域句柄，禁止传入野指针或已销毁的句柄。


## 调用示例

```c
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)

// 所有参与者使用相同配置
HcclCommConfig config;
HcclCommConfigInit(&config);

// 存活rank传入当前通信域句柄；新rank传入NULL
HcclComm oldComm = NULL;
int isNewRank = 0;
uint32_t newRankId = 0;
uint32_t newRankSize = 0;

// rootInfo由存活rank调用HcclCommGetRootInfo获取，并通过业务控制面分发给所有参与者
HcclRootInfo rootInfo = {0};

HcclComm comm = isNewRank ? NULL : oldComm;
HcclComm newComm = NULL;
HCCLCHECK(HcclCommGrow(comm, newRankSize, &rootInfo, newRankId, &config, &newComm));

// 扩容后使用新通信域查询rank信息
uint32_t rankId = 0;
uint32_t rankSize = 0;
HCCLCHECK(HcclGetRankId(newComm, &rankId));
HCCLCHECK(HcclGetRankSize(newComm, &rankSize));
```
