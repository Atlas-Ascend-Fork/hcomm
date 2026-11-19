# HcclCommSuspend

> [!NOTE]说明
> 本接口为预留接口，后续有可能变更，不支持开发者使用。

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas训练系列产品：不支持
<!-- end id5 -->

## 功能说明

当片上内存UCE（Uncorrectable Error）故障时（ACL接口返回ACL_ERROR_RT_DEVICE_MTE_ERROR错误码），可调用本接口将通信域置为挂起状态。

使用此接口挂起通信域，无需退出Host侧进程，后续故障修复后，可调用[HcclCommResume](HcclCommResume.md)接口恢复通信域状态。

## 函数原型

```c
HcclResult HcclCommSuspend(HcclComm comm)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 需要将状态置为挂起的通信域。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 本接口需要与[HcclCommResume](HcclCommResume.md)接口配对使用。
- 本接口不能与集合通信、点对点通信的相关接口并发执行。
- comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。

## 调用示例

```c
// 设备资源初始化
aclInit(NULL);
aclrtSetDevice(devId);

// 创建通信域
HcclComm hcclComm;
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);
HcclCommInitRootInfo(8, &rootInfo, 0, &hcclComm);

// 检测到片上内存UCE故障后，挂起通信域
HcclCommSuspend(hcclComm);

// 故障修复后，恢复通信域
HcclCommResume(hcclComm);

// 销毁通信域
HcclCommDestroy(hcclComm);
aclFinalize();
```
