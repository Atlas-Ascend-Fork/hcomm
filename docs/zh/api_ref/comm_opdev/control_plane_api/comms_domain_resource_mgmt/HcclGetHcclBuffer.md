# HcclGetHcclBuffer

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas推理系列产品：不支持
<!-- end id5 -->

## 功能说明

获取通信域中本端rank的HCCL通信内存，首次调用时将在Device侧进行内存初始化分配，后续调用将复用已分配内存，不会重复初始化。

> [!NOTE]注意
> 返回的HCCL通信内存由hcomm内部管理，调用者严禁释放。

## 函数原型

```c
HcclResult HcclGetHcclBuffer(HcclComm comm, void **buffer, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| buffer | 输出 | HCCL通信内存地址。 |
| size | 输出 | HCCL通信内存大小。内存大小为通信域初始化配置或环境变量HCCL_BUFFSIZE配置值的2倍，默认为400MB。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

如果通信域仅包含一个rank则不会申请HCCL通信内存，出参`buffer`为空指针，`size`为0。

- comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。
- 线程安全：在Ascend 950PR&950DT系列产品上，同一通信域多线程并发调用本接口是安全的；Atlas A2系列与Atlas A3系列产品不支持多线程并发调用。
- 通信域销毁（HcclCommDestroy）不得与本接口并发。

## 调用示例

```c
HcclComm comm;
void *hcclBuffer = nullptr;
uint64_t hcclBufferSize = 1 * 1024;   // 1KB
HcclResult result = HcclGetHcclBuffer(comm, &hcclBuffer, &hcclBufferSize);
```
