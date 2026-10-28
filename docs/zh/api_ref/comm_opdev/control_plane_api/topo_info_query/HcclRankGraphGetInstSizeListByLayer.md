# HcclRankGraphGetInstSizeListByLayer

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

给定通信域和拓扑层级编号，查询该层级下的拓扑实例数量，以及每个实例包含的rank数量。

![拓扑模型](figures/topo_model.png)

以上述拓扑模型为例：

- Layer 0中包含两个拓扑实例，为方便理解，定义拓扑实例ID分别为0和1，每个实例中分别包含3个rank。
- Layer 1中包含1个拓扑实例，实例中包含6个rank。

## 函数原型

```c
HcclResult HcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t **instSizeList, uint32_t *listSize)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| netLayer | 输入 | 拓扑层级编号。 |
| instSizeList | 输出 | 该层级下每个拓扑实例包含的rank数量组成的列表。 |
| listSize | 输出 | instSizeList的大小，即该层级下包含的拓扑实例的数量。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 返回的内存由库内管理，调用者严禁释放。
- 返回结果为通信域内按参数缓存的只读数据。每次调用返回的结果不再变化；同一通信域内后续调用（无论参数是否相同）不会使已返回的结果失效。各通信域独立缓存，互不影响。
- comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。
- 指针仅在快照恢复重建或通信域销毁时失效，且快照恢复/销毁不得与查询并发。
- 不得修改返回的内存；跨线程使用前无需深拷贝。
- 线程安全：在Ascend 950PR&950DT系列产品上，同一通信域多线程并发调用本接口是安全的；Atlas A2系列与Atlas A3系列产品不支持多线程并发调用。
- 通信域销毁（HcclCommDestroy）不得与本接口并发，销毁前须确保所有查询线程已结束对该通信域的所有调用。

## 调用示例

以[功能说明](#功能说明)的拓扑模型为例：

```c
HcclComm comm;
uint32_t *instSizeList;
uint32_t listSize;
HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizeList, &listSize);
// instSizeList=[3,3], listSize=2
HcclRankGraphGetInstSizeListByLayer(comm, 1, &instSizeList, &listSize);
// instSizeList=[6], listSize=1
```
