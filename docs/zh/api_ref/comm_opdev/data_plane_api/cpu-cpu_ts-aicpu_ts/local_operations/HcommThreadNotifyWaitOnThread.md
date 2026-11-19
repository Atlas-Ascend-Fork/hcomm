# HcommThreadNotifyWaitOnThread

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

等待同步信号，该接口会阻塞等待Thread的运行，直到指定的Notify被record完成。Host CPU线程模式下，该接口将等待任务提交至Host CPU线程的任务队列后立即返回，接口返回成功仅表示提交成功。

## 函数原型

```c
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| thread | 输入 | 线程句柄，为通过[HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md)或[HcclThreadAcquireWithConfig](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquireWithConfig.md)接口获取到的threads；Host CPU线程（`THREAD_TYPE_CPU`）仅可由[HcclThreadAcquireWithConfig](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquireWithConfig.md)接口申请。<br>ThreadHandle类型的定义请参见[ThreadHandle](../../../datatype_definition/ThreadHandle.md)。 |
| notifyIdx | 输入 | 需等待的Notify索引。<br>取值范围为：[0, 申请该线程时配置的notifyNumPerThread参数的值)。 |
| timeOut | 输入 | 超时时间，单位：秒。<br>  - 0：表示永久等待；Host CPU线程模式下表示立即超时返回。<br>  - >0：配置的具体超时时间。<br> |

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

- 该接口需要配合[HcommThreadNotifyRecordOnThread](HcommThreadNotifyRecordOnThread.md)使用。

<!-- npu="950" id6 -->
- 在Ascend 950PR&950DT系列产品上，支持AICPU_TS模式下在Device侧调用，也支持以Host CPU线程（[HcclThreadAcquireWithConfig](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquireWithConfig.md)接口申请的`THREAD_TYPE_CPU`线程）在Device侧调用。
- Host CPU线程模式下，该接口将等待任务提交至Host CPU线程的任务队列后立即返回，由Host侧按提交顺序执行，接口返回成功仅表示提交成功；`timeOut`须大于0，为0时立即超时返回，不表示永久等待。
<!-- end id6 -->

- 同一个`ThreadHandle`不支持多线程并发访问。

## 调用示例

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU_TS;
aclrtStream streams[2];
ThreadHandle threads[2];
// 申请2条流，每条流2个Notify
aclrtCreateStream(&streams[0]);
aclrtCreateStream(&streams[1]);
HcclResult result = HcclThreadAcquireWithStream(comm, engine, streams[0], 2, &threads[0]);
result = HcclThreadAcquireWithStream(comm, engine, streams[1], 2, &threads[1]);
uint32_t notifyIdx = 0;
// 发送同步信号
HcommThreadNotifyRecordOnThread(threads[0], threads[1], notifyIdx);
uint32_t timeout = 1;
// 等待同步信号
HcommThreadNotifyWaitOnThread(threads[1], notifyIdx, timeout);
```

<!-- npu="950" id7 -->
在Ascend 950PR&950DT系列产品上，该函数需要编译到Device侧使用：

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
ThreadHandle threads[2];
uint32_t notifyNumPerThread = 2;
HcclThreadAcquire(comm, engine, 2, notifyNumPerThread, threads);

// 其余资源申请
// 拷贝参数并Launch Kernel

// Device侧算法编排
uint32_t notifyIdx = 0;
// 发送同步信号
HcommThreadNotifyRecordOnThread(threads[0], threads[1], notifyIdx);
uint32_t timeout = 1;
// 等待同步信号
HcommThreadNotifyWaitOnThread(threads[1], notifyIdx, timeout);
```
<!-- end id7 -->
