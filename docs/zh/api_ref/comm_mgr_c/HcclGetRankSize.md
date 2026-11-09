# HcclGetRankSize

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
<!-- npu="310p" id4 -->
- Atlas推理系列产品：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas训练系列产品：支持
<!-- end id5 -->

## 功能说明

查询指定通信域的rank总数。

## 函数原型

```c
HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 集合通信操作所在的通信域。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| rankSize | 输出 | 指定集合通信域的rank总数输出地址指针。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。
<!-- npu="950" id6 -->
- 针对在Ascend 950PR&950DT系列产品，同一通信域支持多线程并发调用本接口。
<!-- end id6 -->
<!-- npu="A3,910b" id7 -->
- 针对Atlas A2系列产品与Atlas A3系列产品，不支持多线程并发调用本接口。
<!-- end id7 -->
- 通信域销毁（HcclCommDestroy）不得与本接口并发。

## 调用示例

```c
// 初始化通信域
HcclComm comm;
// 获取通信域内的Rank数量
uint32_t rankSize;
HcclGetRankSize(comm, &rankSize);
```
