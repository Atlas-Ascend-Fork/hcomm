# HcclRankGraphGetEndpointInfo

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

获取指定通信设备的拓扑属性信息。

## 函数原型

```c
HcclResult HcclRankGraphGetEndpointInfo(HcclComm comm, uint32_t rankId, const EndpointDesc *endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void *info)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| rankId | 输入 | 需要查询的端点的所属的rank ID。 |
| endpointDesc | 输入 | EndPoint描述符，为通过[HcclRankGraphGetEndpointDesc](HcclRankGraphGetEndpointDesc.md)接口获取到的“endpointDesc”。 |
| endpointAttr | 输入 | 需要查询的EndPoint属性类型。<br>EndpointAttr类型的定义请参见[EndpointAttr](../../datatype_definition/EndpointAttr.md)。 |
| infoLen | 输入 | 提供的info缓冲区大小（字节）。 |
| info | 输出 | 存储属性信息的输出缓冲区指针。 |

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

// 获取拓扑属性信息
uint32_t rankId = 0; // 需要查询的端点所属的rank ID
EndpointAttrBwCoeff bwCoeff = {0};
uint32_t size = sizeof(EndpointAttrBwCoeff); // 必须等于目标类型大小
HcclRankGraphGetEndpointInfo(comm, rankId, endpointDesc, ENDPOINT_ATTR_BW_COEFF, size, &bwCoeff);
```
