# HcclGetCommAsyncError

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
- Atlas推理系列产品：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas训练系列产品：支持
<!-- end id5 -->

## 功能说明

当集群中存在Device网口通信链路不稳定、出现网络拥塞的情况时，Device日志中会存在“error cqe”的打印，我们称这种错误为“RDMA ERROR CQE”错误。

当前版本，此接口仅支持查询通信域内是否存在“RDMA ERROR CQE”的错误。

> [!NOTE]说明
> 此接口为同步接口，即接口调用后需要等待返回结果。

## 函数原型

```c
HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult *asyncError)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 需要查询是否存在错误信息的通信域。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| asyncError | 输出 | - 0：表示该通信域内无错误发生。<br>  - 21：表示该通信域内发生了“RDMA ERROR CQE”的错误。 |

## 返回值

参见[HcclResult](./data_type_definition/HcclResult.md)类型，接口成功返回HCCL_SUCCESS，失败返回对应错误码。

## 约束说明

- 建立通信域后，才可调用此接口。
- 通信域销毁后，不可调用此接口。
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

// 查询通信域内是否存在异步错误
HcclResult asyncError;
HcclGetCommAsyncError(hcclComm, &asyncError);
if (asyncError == HCCL_E_REMOTE) {
    // 通信域内发生了"RDMA ERROR CQE"错误，进行相应处理
}

// 销毁通信域
HcclCommDestroy(hcclComm);
aclFinalize();
```
