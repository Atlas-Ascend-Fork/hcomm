# HcclStuckDetectGetCounterAddr

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
<!-- npu="910" id4 -->
- Atlas训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas推理系列产品：不支持
<!-- end id5 -->

## 功能说明

本接口在Host侧调用，获取指定通信域卡住检测计数器的Device地址。Host侧只负责获取和传递地址，不能直接解引用这些Device地址；Device侧Kernel通过该地址读取计数器。

计数器语义：

- `headOpCounterAddr`：head计数器地址，算子在Device侧开始执行前加1；
- `tailOpCounterAddr`：tail计数器地址，算子在Device侧执行完成后加1；
- `head - tail`表示当前尚未完成的在途算子数；
- `addOneMemAddr`：指向Device侧常量1的地址，Kernel通过原子加或LocalReduce将head/tail加1。

典型使用方式：

1. Host侧创建并初始化V2通信域，确保`stuck_detection`已开启；
2. Host侧调用本接口，获取`headOpCounterAddr`、`tailOpCounterAddr`、`addOneMemAddr`；
3. Host侧将这三个值写入Kernel入参，并补充计数器大小`counterMemSize = sizeof(int32_t)`和使能标记`isEnableCounter`；
4. Device侧Kernel根据地址对head/tail执行加1，并根据`head - tail`判断通信域是否继续推进。

在hccl仓的AIV/CCU/AICPU接入中，通常按如下方式映射到Kernel入参：

| 本接口输出 | 典型Kernel入参 | Device侧用途 |
| --- | --- | --- |
| `headOpCounterAddr` | `headCountMem` | head计数器地址，算子开始前加1，检测侧读取 |
| `tailOpCounterAddr` | `tailCountMem` | tail计数器地址，算子完成后加1，检测侧读取 |
| `addOneMemAddr` | `addOneMem` | 常量1地址，作为原子加或LocalReduce的源地址 |

地址为0表示检测未开启或地址获取失败，Kernel应跳过计数逻辑。Host侧建议按通信域缓存本接口的结果，避免每次Kernel下发都查询；通信域销毁或Resume后应重新获取。检测阈值应大于正常排队、NotifyWait等耗时，否则可能在正常等待期间误报。

## 函数原型

```c
HcclResult HcclStuckDetectGetCounterAddr(
    HcclComm comm, uint64_t *headOpCounterAddr, uint64_t *tailOpCounterAddr, uint64_t *addOneMemAddr)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。仅支持V2通信域。<br>HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| headOpCounterAddr | 输出 | head计数器Device地址；未开启或获取异常时返回0。 |
| tailOpCounterAddr | 输出 | tail计数器Device地址；未开启或获取异常时返回0。 |
| addOneMemAddr | 输出 | 常量1的Device地址，供计数归约使用；未开启或获取异常时返回0。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：成功返回`HCCL_SUCCESS`，失败返回对应错误码。

- `HCCL_E_PTR`：`comm`或任一输出指针为空，或内部通信域/管理器为空。
- `HCCL_E_NOT_SUPPORT`：传入legacy通信域。
- `HCCL_E_INTERNAL`：通信域信息无效/不可用、计数地址无效或计数器查询过程发生异常。
- `HCCL_E_MEMORY`：计数池分配失败或slot耗尽。
- 检测开关关闭时返回`HCCL_SUCCESS`，三个输出参数均为0。

## 约束说明

- 仅支持V2通信域；legacy通信域返回`HCCL_E_NOT_SUPPORT`。
- 本接口只能在Host侧调用。返回的地址是Device虚拟地址，Host侧不得直接解引用，应通过Kernel入参传给Device侧使用。
- 建议将三个输出放入同一个Kernel入参结构体下发；Device侧读取head/tail时按`int32_t`处理，`addOneMemAddr`仅作为常量1地址传递。
- 即使用户已经拿到非0地址，Host侧也不能用普通内存访问方式读取计数器；计数器由Device侧Kernel或CCU访问。
- 是否启用卡住检测由`HCCL_DFS_CONFIG`的`stuck_detection:on/off`控制，默认开启；取值区分大小写。
- `cluster_heartbeat`关闭时不会建立心跳链路，也不会向对端广播；但`ClusterMonitor`监控线程仍可能启动或复用，接口仍可能返回有效地址并参与本地检测。
- 卡住检测依赖`ClusterMonitor`后台线程采样。当前`rankSize == 1`的通信域不会注册到`ClusterMonitor`；纯单rank进程不会启动检测线程，计数不会被采样。若同设备已有其他多rank通信域启动线程，单rank通信域懒分配的计数可能被该线程采样，但不保证。不能仅凭接口返回有效地址认为检测已生效。
- 返回地址仅在`comm`存活期间有效，通信域销毁后不得继续使用；slot回收后可能被其他通信域复用。
- 不得与`HcclCommDestroy`并发调用。

## 调用示例

### Host侧获取并传递

Host侧获取地址，并作为Kernel入参下发：

```c
// Kernel入参：Host侧填写，Device侧Kernel使用
typedef struct {
    uint64_t headCountMem;
    uint64_t tailCountMem;
    uint64_t addOneMem;
    uint32_t counterMemSize;
    uint32_t isEnableCounter;
} StuckDetectKernelArgs;

// 前置条件：comm已由通信域创建接口初始化为V2通信域
HcclComm comm = nullptr; // 占位：替换为已初始化的V2通信域句柄
if (comm == nullptr) {
    return HCCL_E_PTR;
}
uint64_t headOpCounterAddr = 0;
uint64_t tailOpCounterAddr = 0;
uint64_t addOneMemAddr = 0;

HcclResult ret = HcclStuckDetectGetCounterAddr(
    comm, &headOpCounterAddr, &tailOpCounterAddr, &addOneMemAddr);
if (ret != HCCL_SUCCESS) {
    // 错误处理
    return ret;
}

StuckDetectKernelArgs args = {};
args.headCountMem = headOpCounterAddr;
args.tailCountMem = tailOpCounterAddr;
args.addOneMem = addOneMemAddr;
args.counterMemSize = sizeof(int32_t);
args.isEnableCounter =
    (headOpCounterAddr != 0 && tailOpCounterAddr != 0 && addOneMemAddr != 0) ? 1 : 0;

// 将args作为Kernel入参下发；Host侧不要直接解引用这些地址
```

### Device侧使用

Device侧Kernel读取计数器（伪代码）：

```c
// args为Host侧下发的Kernel入参
if (args->isEnableCounter != 0) {
    volatile int32_t *head = (volatile int32_t *)(uintptr_t)args->headCountMem;
    volatile int32_t *tail = (volatile int32_t *)(uintptr_t)args->tailCountMem;
    int32_t headValue = *head;
    int32_t tailValue = *tail;
    // 根据head - tail判断通信域是否继续推进
}

// args->addOneMem指向常量1，作为head/tail计数加1的源地址
```
