# HcclGroupEnqueueOp

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
<!-- npu="310p" id4 -->
- Atlas推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas训练系列产品：不支持
<!-- end id5 -->

## 功能说明

Send/Receive算子入口检测group状态，group开启时将算子信息按序加入队列缓存。

须在[HcclGroupStart](HcclGroupStart.md)之后调用；[HcclGroupEnd](HcclGroupEnd.md)时按序出队，并将[HcclOpDesc](./data_type_definition/HcclOpDesc.md)中的Send/Receive算子信息转换为[HcclSendRecvItem](./data_type_definition/HcclSendRecvItem.md)数组后，调用入队时指定的批量执行函数统一执行，实现group内多个点对点通信任务的合并下发。

## 函数原型

```c
HcclResult HcclGroupEnqueueOp(HcclComm comm, const HcclOpDesc* opDesc, aclrtStream stream, void* func)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄，类型定义请参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| opDesc | 输入 | 算子描述符指针，仅支持Send/Receive算子，可通过[HcclOpDescInit](HcclOpDescInit.md)初始化后填充p2p成员，类型定义请参见[HcclOpDesc](./data_type_definition/HcclOpDesc.md)。 |
| stream | 输入 | 本rank所使用的stream，HcclGroupEnd时批量执行以同一通信域内首条任务的stream为准。 |
| func | 输入 | HcclGroupEnd时批量执行的函数指针，入队任务将按序转换为sendRecvInfo数组传入。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。主要失败场景：

- HCCL_E_INTERNAL：group未开启（未调用HcclGroupStart），或group内入队任务总数达到上限2048。
- HCCL_E_PTR：参数为空指针，或通信域无队列实例（即非V2通信域）。
- HCCL_E_PARA：opDesc中的算子类型不是Send/Receive。

## 约束说明

- 须在HcclGroupStart与HcclGroupEnd之间调用，group未开启时入队失败。
- opDesc的p2p.cmdType仅支持HCCL_CMD_SEND和HCCL_CMD_RECEIVE，类型定义请参见[HcclCMDType](./data_type_definition/HcclCMDType.md)。
- 仅支持V2通信域（Ascend 950PR&950DT系列），非V2通信域返回HCCL_E_PTR。
- group内入队的算子任务总数上限为2048（跨通信域累计），超出后返回HCCL_E_INTERNAL。
- func的签名须为`HcclResult (*)(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)`。
- 同一通信域内多次入队传入的func须为同一函数，否则HcclGroupEnd批量执行时返回HCCL_E_PARA。
- 同一通信域内多次入队可传入不同stream，但批量执行时仅首条任务入队时指定的stream生效。
- 在一个组调用中，通信域管理、集合通信、点对点通信类型的接口不可混用。

## 调用示例

```c
// 需包含头文件：hccl/hccl_comm.h（HcclGroupStart/HcclGroupEnqueueOp/HcclGroupEnd）、
// hccl/hccl_launch.h（HcclOpDesc/HcclOpDescInit）、acl/acl.h（aclrtStream）
// 返回值检查宏
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)

// HcclGroupEnd时批量执行的函数，签名须满足约定
HcclResult BatchSendRecvExec(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum,
    HcclComm comm, aclrtStream stream)
{
    // 批量下发sendRecvInfo数组中的Send/Receive通信任务
    // （此处为用户自己的批量下发实现）
    return HCCL_SUCCESS;
}

// group内合并下发多个Send/Receive任务
HcclResult GroupEnqueueExample(HcclComm hcclComm, aclrtStream stream, void* sendBuf, void* recvBuf,
    uint64_t count, uint32_t peerRank)
{
    HcclOpDesc sendDesc;
    HcclOpDescInit(&sendDesc);
    sendDesc.p2p.cmdType = HCCL_CMD_SEND;
    sendDesc.p2p.buffer = sendBuf;
    sendDesc.p2p.count = count;
    sendDesc.p2p.dataType = HCCL_DATA_TYPE_FP32;
    sendDesc.p2p.remoteRank = peerRank;

    HcclOpDesc recvDesc;
    HcclOpDescInit(&recvDesc);
    recvDesc.p2p.cmdType = HCCL_CMD_RECEIVE;
    recvDesc.p2p.buffer = recvBuf;
    recvDesc.p2p.count = count;
    recvDesc.p2p.dataType = HCCL_DATA_TYPE_FP32;
    recvDesc.p2p.remoteRank = peerRank;

    HCCLCHECK(HcclGroupStart());
    HCCLCHECK(HcclGroupEnqueueOp(hcclComm, &sendDesc, stream, (void*)BatchSendRecvExec));
    HCCLCHECK(HcclGroupEnqueueOp(hcclComm, &recvDesc, stream, (void*)BatchSendRecvExec));
    HCCLCHECK(HcclGroupEnd());

    return HCCL_SUCCESS;
}
```
