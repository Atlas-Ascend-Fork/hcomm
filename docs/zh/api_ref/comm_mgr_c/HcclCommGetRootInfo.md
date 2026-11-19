# HcclCommGetRootInfo

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

从已有通信域获取供[HcclCommGrow](HcclCommGrow.md)使用的rootInfo。

如果当前通信域的root有效，本接口返回该root的信息；如果当前root已失效，例如root在缩容时被剔除，本接口会触发root重选并创建新的root server。

本接口与[HcclGetRootInfo](HcclGetRootInfo.md)的用途不同：[HcclGetRootInfo](HcclGetRootInfo.md)用于创建新通信域时生成root信息，本接口用于从已有通信域获取扩容所需的root信息。

## 函数原型

```c
HcclResult HcclCommGetRootInfo(HcclComm comm, HcclRootInfo *rootInfo)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 已有通信域句柄。必须由rootInfo方式创建，且为当前最新一代通信域。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| rootInfo | 输出 | 供扩容使用的root信息。获取后需分发给所有扩容参与者。<br>HcclRootInfo类型的定义可参见[HcclRootInfo](./data_type_definition/HcclRootInfo.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

| 返回值 | 说明 |
| --- | --- |
| HCCL_SUCCESS | 获取rootInfo成功。 |
| HCCL_E_PTR | `comm`或`rootInfo`为空指针，或通信域内部对象异常。 |
| HCCL_E_PARA | 通信域派生代数已溢出，无法生成新的rootInfo。 |
| HCCL_E_INTERNAL | root重选、root server创建或root信息获取失败。 |

## 约束说明

- 仅支持Ascend 950PR&950DT系列产品。
- 本接口应在扩容前调用，由业务侧选择一个存活rank执行，并将获取到的`rootInfo`分发给所有存活rank和新rank。
- 如果当前root有效，建议由root所在rank调用；如果root已失效，必须由一个存活rank调用以触发root重选。
- 不能由多个rank并发触发root重选。
- 本接口不能与同一通信域上的集合通信、点对点通信、扩缩容、通信域销毁等操作并发执行。
- 本接口不支持多线程并发调用，调用方须保证不同线程对本接口的调用串行执行。
- 默认不产生编译期告警。若定义`HCCL_ENABLE_EXPERIMENTAL_WARNING`开启试验特性告警，且编译选项同时开启`-Werror`（或MSVC `/WX`），告警会导致编译失败。请仅在调用点附近局部抑制告警，并在调用后恢复告警设置。例如，GCC/Clang可使用`#pragma GCC diagnostic ignored "-Wattribute-warning"`，MSVC可使用`#pragma warning(disable: 4996)`。
- `comm`必须为有效的通信域句柄，禁止传入空指针、野指针或已销毁的句柄。


## 调用示例

```c
// 试验特性：本接口后续版本可能调整或移除，不承诺向后兼容。
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)

// 已有通信域，由业务按实际场景设置
HcclComm comm = NULL;
HcclRootInfo rootInfo = {0};

// 仅由一个存活rank调用
HCCLCHECK(HcclCommGetRootInfo(comm, &rootInfo));

// 业务侧通过控制面通道将rootInfo分发给所有扩容参与者。
// 所有参与者调用HcclCommGrow时必须传入相同的rootInfo。
```
