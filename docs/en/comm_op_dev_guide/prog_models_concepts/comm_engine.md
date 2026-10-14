# Communication Engine

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-24T07:09:18.829Z pushedAt=2026-09-28T05:59:29.463Z -->

HCCL can use different communication engines to implement communication operators. Different communication engines are suitable for different scenarios. The comparison results are as follows.

**Table 1** Applicable scenarios of different communication engines

| Communication Engine | Advantage | Constraint | Applicable Scenario | Supported Product Model |
| --- | --- | --- | --- | --- |
| AI CPU+TS | Does not occupy compute cores, offers high communication efficiency, and is suitable for large-data, high-bandwidth scenarios. | Has high static communication overhead and is not suitable for small-data communication scenarios. | High-bandwidth communication scenarios | Ascend 950PR&950DT products<br>Atlas A3 products |
| Host CPU+TS | Does not occupy compute cores. | Has high dispatch overhead that increases linearly with the number of tasks. | N/A | Atlas A2 products |
| AIV | Low latency. | Communication occupies Vector compute cores, and multiple Vector compute cores are required to fully utilize the communication bandwidth. Communication operators compete with compute operators for compute core resources, which may cause mutual interference. | Low-latency communication scenarios | Ascend 950PR&950DT products<br>Atlas A3 products (only intra-supernode communication scenarios supported)<br>Atlas A2 products (only single-node communication scenarios supported) |
| CCU | Reduces memory access bandwidth and compute core usage.<br>High bandwidth and low latency. | Limited by on-chip resources, the number of supported communicators is limited. | High-bandwidth, low-latency communication scenarios | Ascend 950PR&950DT products |

The following describes the task execution process of each communication engine.

## AI CPU+TS

The AI CPU submits communication operation-related tasks to the Task Scheduler (TS).

1. The host submits an AI CPU kernel to the task queue.
2. After being scheduled by the task scheduler, the AI CPU kernel is dispatched to the AI CPU for execution.
3. The AI CPU submits communication tasks to the task queue.
4. The communication task submitted by the AI CPU is scheduled by the scheduler to the executor for execution.

**Figure 1** AI CPU+TS scheduling  
![](figures/aicpu_ts_schedule.png "AI-CPU+TS-scheduling")

## Host CPU+TS

The host CPU submits communication operation-related tasks to the Task Scheduler (TS) on the device.

1. The host submits various operations during the communication process (including memory copy and synchronization operations) to the task queue.
2. The scheduler dispatches the tasks in the task queue to the corresponding executors for execution.

**Figure 2**  Host CPU+TS scheduling  
![](figures/hostcpu_ts_schedule.png "Host-CPU+TS-scheduling")

## AIV

The execution logic and operation steps of the communication operator are executed by the Vector Core.

1. The host submits an AIV kernel to the task queue.
2. The AIV kernel is scheduled by the scheduler and then sent to the Vector Core for execution.
3. The Vector Core can use different protocols to complete data movement.

**Figure 3**  AIV communication  
![](figures/aiv_communication.png "AIV-communication")

## CCU

The communication operator is executed by the Collective Communication Unit (CCU), as shown in the figure.

1. The host dispatches a CCU instruction sequence (consisting of instructions recognizable by the CCU) to the CCU instruction space, and submits the corresponding CCU kernel task to the task queue.
2. The CCU kernel is dispatched by the scheduler and then sent to the CCU for execution.
3. The CCU executes the corresponding instruction stream and uses Unified Remote Memory Access (URMA) to complete data movement.

**Figure 4** CCU acceleration  
![](figures/ccu_communication.png "CCU-acceleration")
