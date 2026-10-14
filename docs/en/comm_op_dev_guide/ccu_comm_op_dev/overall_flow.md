# Overall Process

<!-- md-trans-meta sourceCommit=4b3acc1183ff175f340b0421dbe1faf9a723a585 translatedAt=2026-08-11T07:16:33.752Z pushedAt=2026-08-20T11:39:14.572Z -->

The development process of HCCL communication operators is as follows:

![](figures/ccu_dev_flow.png)

1. **Defining the operator API**: Define the communication operator API based on its functionality.
2. **Querying topology information**: Obtain information such as the network hierarchy and number of nodes in the communicator.
3. **(Optional) Selecting algorithms**: To maximize communication performance, different algorithms are often used for different topologies and data sizes. Therefore, the optimal communication algorithm needs to be selected based on topology information.
4. **Creating resources**: Allocate resources such as memory, threads, Notify, and channels required by the communication algorithm, and complete CCU kernel registration and resource allocation.
5. **Dispatching the operator**: Dispatch the operator kernel to the communication engine for execution.
6. **Executing the algorithms**: Use communication resources to synchronize and transfer data between each rank through the CCU, completing the collective communication operation.
