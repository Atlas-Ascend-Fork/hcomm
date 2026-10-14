# Overall Process

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-11T07:10:51.370Z pushedAt=2026-08-20T11:39:14.566Z -->

The development process of HCCL communication operators is as follows:

![AIV communication operator development process](figures/aiv_dev_flow.png)

1. **Defining the operator API**: Define the communication operator API based on the function of the communication operator.
2. **Querying topology information**: Obtain information such as the network hierarchy and number of nodes in the communicator.
3. **(Optional) Selecting algorithms**: To maximize communication performance, different algorithms are often used for different topologies and data sizes. Therefore, select the optimal communication algorithm based on topology information.
4. **Creating resources**: Allocate the memory, thread, Notify, channel, and other resources required by the communication algorithm.
5. **Scheduling tasks**: Use communication resources to coordinate each rank for synchronization and data movement, completing the collective communication operation.
6. **Dispatching the operator**: Dispatch the operator kernel to the communication engine for execution.
