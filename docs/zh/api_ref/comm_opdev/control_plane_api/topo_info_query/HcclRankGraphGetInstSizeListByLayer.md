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

<!-- npu="950" id6 -->
当该层级建立了虚拟分组视图时，返回各虚拟分段组包含的rank数量列表，`listSize`为虚拟组数量。虚拟分段组是在每个原拓扑实例内，将rank按rank ID升序排列后依据连续性划分而成，不跨实例合并；返回列表按各组起始rank ID升序排列。未建立虚拟分组视图的层级保持原有拓扑实例语义。

例如，某中间层的两个拓扑实例分别包含rank `{0,1,2,3,8,9,10,11}`和`{4,5,6,7,12,13,14,15}`。建立虚拟分组视图后，返回`instSizeList=[4,4,4,4]`、`listSize=4`，而非原拓扑实例对应的`[8,8]`、`2`。
<!-- end id6 -->

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
| listSize | 输出 | instSizeList的大小，即该层级下包含的拓扑实例数量。 |

<!-- npu="950" id7 -->
当该层级建立了虚拟分组视图时，`instSizeList`为各连续虚拟分段组包含的rank数量列表，`listSize`为虚拟组数量。
<!-- end id7 -->

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

<!-- npu="950" id8 -->
- 在Ascend 950PR&950DT系列产品上，通过RootInfoDetect路径构建通信域rank graph时启用虚拟分组策略，通过rank table文件路径构建时不启用。子通信域继承该策略。快照包含分组策略时，恢复时保留该策略；旧版本保存的快照不含分组策略，恢复时禁用虚拟分组，返回原拓扑实例大小列表。
- 当前虚拟分组策略跳过当前rank加入的最低层级，并以物理超节点层为分段上界：不超过上界的其余已加入层级（含上界层本身）均判断是否需要建立虚拟分组，超过上界的层级保持原拓扑实例语义。仅当该层至少一个拓扑实例内存在不连续的rank ID时，才为该层建立虚拟分组视图。查询最低层级、超过上界的层级，或该层各实例内rank ID均连续时，仍返回原拓扑实例大小列表。当前rank仅加入一个层级时，也返回原拓扑实例大小列表。
- 虚拟分组视图在rank graph初始化完成前构建，初始化完成后不再重建。下述内存管理、结果复制时效及comm有效性约束同样适用于虚拟分组结果。
<!-- end id8 -->

- 返回的内存由库内管理，调用者严禁释放。
- 应及时复制返回数据，同一通信域重复调用可能使前次结果失效。
- comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。

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
