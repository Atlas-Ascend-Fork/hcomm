# HcclEngineCtxGet

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

指定通信域和通信引擎，通过通信引擎上下文标签获取对应的通信引擎上下文。

## 函数原型

```c
HcclResult HcclEngineCtxGet(HcclComm comm, const char *ctxTag, CommEngine engine, void **ctx, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| ctxTag | 输入 | 通信引擎上下文标签，最大字符长度为HCCL_RES_TAG_MAX_LEN。<br>const uint32_t HCCL_RES_TAG_MAX_LEN = 255; |
| engine | 输入 | 通信引擎类型。 |
| ctx | 输出 | 通信引擎上下文句柄。 |
| size | 输出 | 通信引擎上下文对应内存大小。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。
- 线程安全：同一通信域多线程并发调用本接口是安全的（内部已加锁保护）。
- 返回的ctx指针在HcclEngineCtxDestroy后不可再使用；ctx使用期间不得销毁通信域。

## 调用示例

```c
HcclComm comm;
uint64_t size = 0;
void *ctx = nullptr;
const char *ctxTag = "ctxTag";
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
HcclResult ret = HcclEngineCtxGet(comm, ctxTag, engine, &ctx, &size);
```
