# HcclRankGraphGetRanksByTopoInst

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

给定通信域和拓扑层级编号，查询当前rank对应的指定拓扑实例中包含的rank信息。

## 函数原型

```c
HcclResult HcclRankGraphGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, uint32_t **ranks, uint32_t *rankNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| netLayer | 输入 | 拓扑层级编号。 |
| topoInstId | 输入 | 拓扑实例ID。（拓扑文件中存在的） |
| ranks | 输出 | 对应拓扑实例中包含的rank列表。 |
| rankNum | 输出 | 列表数量。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 启用虚拟分组且查询层存在虚拟分组时，仅返回指定拓扑实例与当前rank所在虚拟分组的成员交集。交集为空时返回HCCL_E_NOT_FOUND，调用者应先检查返回码再使用输出参数。
- 返回的内存由库内管理，调用者严禁释放。
- 应及时复制返回数据，同一通信域重复调用可能使前次结果失效。
- comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。

## 调用示例

```c
 //8卡通信域，同一个8p Mesh
HcclComm comm;
uint32_t netLayer = 0;
uint32_t topoInstId = 0;
uint32_t *ranks;
uint32_t rankNum;
HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInstId, &ranks, &rankNum);
 // ranks = [0,1,2,…,7],  rankNum=8
```
