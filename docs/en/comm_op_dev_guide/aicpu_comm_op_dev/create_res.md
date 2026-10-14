# Creating Resources

<!-- md-trans-meta sourceCommit=73cabe45b7e2f3d9616f1af5c4030feff964f98d translatedAt=2026-09-24T07:04:26.269Z pushedAt=2026-09-28T05:59:29.453Z -->

## Communication Resource Computation

Before task orchestration, a communication operator needs to compute resources based on the algorithm's concurrency mode and communication objects. AI CPU communication primarily uses the following resources:

- **Context**: Stores information about resources such as threads, channels, and communication memory. After resource creation on the host, the context is serialized and copied to the device for deserialization by the AI CPU kernel.
- **Thread**: Expresses the serial and concurrent relationships of tasks within a single rank. Tasks on the same thread are executed sequentially, while tasks on different threads can be executed concurrently. An algorithm typically has one primary thread and several secondary threads.
- **Thread Notify**: Used for synchronization between the primary and secondary threads within the same rank.
- **Channel**: Connects the local rank with a remote rank for cross-rank data transfer and synchronization.
- **Channel Notify**: A synchronization resource belonging to a channel, used to notify the remote end of states such as data readiness and transfer completion.
- **Communication memory**: Includes the local CCL buffer and the remote CCL buffer obtained through the channel.

Thread Notify and channel Notify have different scopes. They should be counted separately when computing resources, rather than using a single total number of Notify resources.

## Full-Mesh Resource Example

Take the single-server 4-device full-mesh topology of the current custom AllGather AI CPU as an example. Each rank communicates directly with the other three ranks.

![Mesh Algorithm Hardware Topology Example](./figures/mesh.png)

### Thread and Thread Notify

In the example, each concurrent communication path is assigned an algorithm thread, so each rank requires three algorithm threads.

- One primary thread, responsible for local copy and coordinating secondary threads.
- Two secondary threads, responsible for the remaining concurrent communication paths.

Primary-secondary thread synchronization is organized in pairs:

1. In the start phase, the primary thread notifies the two secondary threads to begin execution.
2. In the end phase, the two secondary threads notify the primary thread that they have completed.

Therefore, from the logical requirement perspective:

| Thread Type | Quantity | Notify Resources per Thread | Notify Subtotal |
| --- | ---: | ---: | ---: |
| Primary thread | 1 | 2 | 2 |
| Secondary thread | 2 | 1 | 2 |
| **Total** | **3** | - | **4** |

After generalization, if the algorithm has a total of `T` threads, there is 1 primary thread and `T - 1` secondary threads.

- The primary thread requires `T - 1` Notify resources, each corresponding to a secondary thread.
- Each secondary thread requires one Notify resource for synchronizing with the primary thread.
- The total number of thread Notify resources logically required by the algorithm is `2 × (T - 1)`.

The resource API may uniformly allocate Notify resources based on the maximum number of Notify resources for each thread. Therefore, the physically allocated Notify resources may exceed the logically used Notify resources described above. Use the Notify indexes actually used by the algorithm as the reference, and avoid mistaking the uniform allocation upper limit of the API as meaning that every thread uses the same number of Notify resources.

### Channel and Channel Notify

In a 4-device mesh, each rank needs to connect to the other three ranks, so three channels are required. In this example, three Notify resources are allocated for each channel:

- `ACK`: Both parties confirm that the data is ready.
- `DATA_SIGNAL`: Both parties confirm that the current round of data transmission is complete.
- `FIN_ACK`: A reserved completion confirmation Notify, facilitating subsequent extension of a complete end synchronization.

Therefore, the number of channel Notify resources allocated for each rank is `3 channels × 3 = 9`. The current task orchestration primarily uses the first two Notify indexes, and the resource allocation is still based on the channel description in the code.

### Host and AI CPU Control Synchronization

In addition to the algorithm threads, the sample also creates a pair of threads for synchronization control between the host and the AI CPU kernel:

- Before the kernel is delivered, the host thread notifies the AI CPU control thread that execution can begin.
- The AI CPU control thread notifies the host thread after completing task orchestration.
- Each of the two threads uses one control Notify resource, which is not counted in the four logical Notify resources of the algorithm's primary and secondary threads.

The resource summary for the single-server 4-device sample is as follows.

| Communication Resource | Quantity per Rank | Description |
| --- | ---: | --- |
| Algorithm Thread | 3 | One primary thread and two secondary threads |
| Algorithm Thread Notify | 4 | Two for the primary thread and one for each secondary thread |
| Channel | 3 | Connected to the other three ranks respectively |
| Channel Notify | 9 | Three Notify resources for each channel |
| Host/AI CPU Control Notify | 2 | One for the host control thread and one for the AI CPU control thread |

## Sample Code

The following code snippet shows how the main resources are computed in the custom AllGather sample of the current AI CPU:

1. Compute the algorithm threads and thread Notify resources.

    ```cpp
    uint32_t threadNum = rankSize > 1 ? rankSize - 1 : 1;
    resource.slaveThreadNum = threadNum - 1;
    resource.notifyNumOnMainThread = resource.slaveThreadNum;
    resource.notifyNumPerThread =
        std::vector<uint32_t>(resource.slaveThreadNum, 1);
    ```

2. Allocate the algorithm threads. The API allocates thread Notify resources uniformly based on the maximum number of Notify resources for each thread. The sample reserves one additional Notify resource for synchronization control.

    ```cpp
    uint32_t maxNotifyNum = resource.notifyNumOnMainThread;
    std::vector<ThreadHandle> threads(threadNum);
    HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS,
                      threadNum, maxNotifyNum + 1, threads.data());
    ```

3. Establish a channel for each remote rank and set the number of channel Notify resources.

    ```cpp
    constexpr uint32_t channelNotifyNum = 3;
    HcclChannelDesc desc;
    HcclChannelDescInit(&desc, 1);
    desc.remoteRank = remoteRank;
    desc.notifyNum = channelNotifyNum;
    desc.channelProtocol = selectedProtocol;
    HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS,
                       &desc, 1, &channelHandle);
    ```

4. Obtain the local and remote communication memory, and write them into the context resource.

    ```cpp
    void *localAddr = nullptr;
    uint64_t localSize = 0;
    HcclGetHcclBuffer(comm, &localAddr, &localSize);

    void *remoteAddr = nullptr;
    uint64_t remoteSize = 0;
    HcclChannelGetHcclBuffer(comm, channelHandle,
                             &remoteAddr, &remoteSize);
    ```

After the resources are created, the host serializes the context and copies it to the device. After the AI CPU kernel is started, it deserializes the context and then uses the threads, Notify resources, channels, and communication memory within it for task orchestration.
