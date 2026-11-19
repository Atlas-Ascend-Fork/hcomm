# HcclThreadAcquireWithConfig

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

基于通信域和线程配置获取通信线程，并为每条通信线程分配指定数量的同步资源（Notify）。相关概念可参见[通信算子开发指南-并发模型](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md)章节。

> [!NOTE]说明
> 与[HcclThreadAcquire](./HcclThreadAcquire.md)相比，该接口支持通过ThreadConfig结构体逐线程配置同步资源（Notify）数量，适用于不同线程需要不同Notify数量的场景；还支持申请Host CPU线程（`type = THREAD_TYPE_CPU`）。

## 函数原型

```c
HcclResult HcclThreadAcquireWithConfig(HcclComm comm, CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig *config, ThreadHandle *threads)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| engine | 输入 | 通信引擎类型。<br>CommEngine类型的定义可参见[CommEngine](../../datatype_definition/CommEngine.md)。 |
| threadNum | 输入 | 通信线程数量。 |
| type | 输入 | 线程类型，支持THREAD_TYPE_TS与THREAD_TYPE_CPU（Host CPU线程仅支持在Ascend 950PR&950DT系列产品上申请）。<br>ThreadType类型的定义可参见[ThreadType](../../datatype_definition/ThreadType.md)。 |
| config | 输入 | 线程配置信息数组，数组长度须与threadNum一致。调用前须使用[ThreadConfigInit](./ThreadConfigInit.md)接口进行参数初始化。 <br>ThreadConfig类型的定义可参见[ThreadConfig](../../datatype_definition/ThreadConfig.md)。|
| threads | 输出 | 返回的通信线程句柄。需传入threadNum大小的ThreadHandle类型数组；Host CPU线程的句柄可直接作为kernel参数下发至Device侧使用。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../datatype_definition/ThreadHandle.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口返回值说明如下：

## 约束说明

1. 返回的通信线程与同步资源由库内管理，调用者严禁释放。

2. 当前各产品形态支持的CommEngine范围：

   <!-- npu="950" id6 -->
   - Ascend 950PR&950DT系列产品：
     - COMM_ENGINE_CPU
     - COMM_ENGINE_AICPU
   <!-- end id6 -->

   <!-- npu="A3" id7 -->
   - Atlas A3系列产品：
     - COMM_ENGINE_CPU
     - COMM_ENGINE_AICPU
   <!-- end id7 -->

   <!-- npu="910b" id8 -->
   - Atlas A2系列产品：
     - COMM_ENGINE_CPU
     - COMM_ENGINE_AICPU
   <!-- end id8 -->

3. 该接口不支持COMM_ENGINE_AIV、COMM_ENGINE_CCU、COMM_ENGINE_CPU_TS、COMM_ENGINE_AICPU_TS通信引擎。

   > 如需创建TS类型线程，需传入COMM_ENGINE_CPU或COMM_ENGINE_AICPU通信引擎，并将`type`配置为THREAD_TYPE_TS类型。

4. comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。

5. 同一通信域内以相同engine与type重复申请时，返回已创建的通信线程句柄，不重复创建；需要多条线程时应一次申请。

6. 同一通信域内后续申请携带的notifyNumPerThread大于线程当前Notify数量时触发扩容，Notify容量只增不减。

<!-- npu="950" id9 -->
7. Host CPU线程（`type = THREAD_TYPE_CPU`）仅支持在Ascend 950PR&950DT系列产品上申请，且须使用`COMM_ENGINE_AICPU`引擎。
8. Host CPU线程上的数据面任务由Host侧按提交顺序执行，须由单一编排kernel顺序提交；任务执行失败后该线程停止执行后续已提交任务，错误经任务异常机制上报。
9. Host CPU线程扩容后其Notify状态整体清零，应在该线程无在途任务时扩容。
<!-- end id9 -->

## 调用示例

创建线程资源示例如下：

```c
// 通信域句柄
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU;
uint32_t threadNum = 5;
ThreadConfig configs[5];
ThreadConfigInit(configs, threadNum);
for (uint32_t i = 0; i < threadNum; i++) {
    configs[i].notifyNumPerThread = 2;
}
ThreadHandle threads[5];
HcclThreadAcquireWithConfig(comm, engine, threadNum, THREAD_TYPE_TS, configs, threads);
```

不同线程配置不同Notify数量的示例如下：

```c
// 通信域句柄
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU;
uint32_t threadNum = 3;
ThreadConfig configs[3];
ThreadConfigInit(configs, threadNum);
configs[0].notifyNumPerThread = 2;
configs[1].notifyNumPerThread = 4;
configs[2].notifyNumPerThread = 1;
ThreadHandle threads[3];
HcclThreadAcquireWithConfig(comm, engine, threadNum, THREAD_TYPE_TS, configs, threads);
```

申请Host CPU线程示例如下：

```c
// 通信域句柄
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU;
ThreadConfig config;
ThreadConfigInit(&config, 1);
config.notifyNumPerThread = 2;
ThreadHandle dpuThread;
HcclThreadAcquireWithConfig(comm, engine, 1, THREAD_TYPE_CPU, &config, &dpuThread);
// 将dpuThread作为kernel参数下发编排kernel后，在Device侧以Host CPU线程提交数据面任务。
```
