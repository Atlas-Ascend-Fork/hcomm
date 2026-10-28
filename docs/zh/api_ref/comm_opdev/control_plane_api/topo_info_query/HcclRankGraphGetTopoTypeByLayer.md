# HcclRankGraphGetTopoTypeByLayer

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

给定通信域和拓扑层级编号，返回本rank所在拓扑层级中的拓扑类型。

![拓扑模型](figures/topo_model.png)

以上述拓扑模型为例：

- Layer 0中包含两个拓扑实例，为方便理解，定义拓扑实例ID分别为0和1。ID为0的拓扑类型为1DMesh，ID为1的拓扑类型为Clos。
- Layer1中包含1个拓扑实例，拓扑类型为Clos。

## 函数原型

```c
HcclResult HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, CommTopo *topoType)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| netLayer | 输入 | 拓扑层级编号。 |
| topoType | 输出 | 拓扑类型，包括1DMesh、Clos、自定义等。<br>CommTopo类型的定义可参见[CommTopo](../../datatype_definition/CommTopo.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。
- 线程安全：在Ascend 950PR&950DT系列产品上，同一通信域多线程并发调用本接口是安全的；Atlas A2系列与Atlas A3系列产品不支持多线程并发调用。
- 通信域销毁（HcclCommDestroy）不得与本接口并发。

## 调用示例

以[功能说明](#功能说明)的拓扑模型为例。

对于rank0：

```c
HcclComm comm;
CommTopo topoType;
HcclRankGraphGetTopoTypeByLayer(comm, 0, &topoType);  
// Layer0的topoType=1 (1DMesh)
HcclRankGraphGetTopoTypeByLayer(comm, 1, &topoType);  
// Layer1的topoType=0 (Clos)
```

对于rank3：

```c
HcclComm comm;
CommTopo topoType;
HcclRankGraphGetTopoTypeByLayer(comm, 0, &topoType);  
// Layer0的topoType=0 (Clos)
HcclRankGraphGetTopoTypeByLayer(comm, 1, &topoType);  
// Layer1的topoType=0 (Clos)
```
