# HcclRankGraphGetEndpointDesc

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

获取拓扑实例的EndPoint描述列表。

## 函数原型

```c
HcclResult HcclRankGraphGetEndpointDesc(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t *descNum, EndpointDesc *endpointDesc)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| layer | 输入 | 拓扑层级编号。 |
| topoInstId | 输入 | 拓扑实例ID。 |
| descNum | 输入/输出 | 作为输出时，实际获取的通信设备描述数量。<br>作为输入时，需要等于[HcclRankGraphGetEndpointNum](HcclRankGraphGetEndpointNum.md)接口的输出“num”的值。 |
| endpointDesc | 输出 | EndPoint描述列表，需要调用方分配内存。<br>EndpointDesc类型的定义请参见[EndpointDesc](../../datatype_definition/EndpointDesc.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。
- 线程安全：在Ascend 950PR&950DT系列产品上，同一通信域多线程并发调用本接口是安全的；Atlas A2系列与Atlas A3系列产品不支持多线程并发调用。
- 通信域销毁（HcclCommDestroy）不得与本接口并发。

## 调用示例

```c
// 通信域句柄
HcclComm comm;

// 获取EndPoint数量
uint32_t layer = 0;
uint32_t topoInstId = 0;
uint32_t num = 0;
HcclRankGraphGetEndpointNum(comm, layer, topoInstId, &num);

// 获取EndPoint描述列表
uint32_t descNum = num;
EndpointDesc endpointDesc[descNum];
HcclRankGraphGetEndpointDesc(comm, layer, topoInstId, &descNum, endpointDesc);
```
