# Creating Resources

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-24T07:05:45.537Z pushedAt=2026-09-28T05:59:29.456Z -->

## Communication Resource Computation

Communication operators rely on underlying hardware communication resources during execution. Therefore, before scheduling and dispatching operator tasks, you need to first allocate the required communication resources. The communication resources required by AIV communication operators are mainly classified into three types:

- **Vector Core**: Used to express concurrency within a single rank. During algorithm orchestration, different ranks or different data blocks can be split and processed in parallel on multiple compute cores. Multiple AI Cores share the same code logic, and the only difference between instances running on each compute core is the core ID (index). Each core determines its algorithm branch flow based on its own index.
- **Notify**: Used to implement synchronization mechanisms, including synchronization between different Vector Cores within a rank and synchronization between ranks. Since hardware capabilities cannot be called within an AI Core to implement inter-rank synchronization, a software algorithm is required to simulate the synchronization function, which is called soft synchronization. The Notify used for soft synchronization is essentially a block of on-chip memory.
- **Channel**: Used for cross-rank data transfer.

Different communication algorithms often require different types and quantities of communication resources. The following uses a single-server 4-device topology with the AIV communication engine as an example to explain the logic for calculating the quantity of communication resources required by the fully-connected Mesh communication algorithm.

The hardware topology of the Mesh algorithm is shown in the following figure, where each rank directly communicates with all other ranks.

![Mesh algorithm hardware topology](figures/mesh.png)

In the preceding networking setup, the number of communication resources that need to be allocated or occupied by the fully-connected Mesh communication algorithm is as follows:

- Vector Core: Each rank needs to communicate with all other ranks. The communication tasks on each rank (including the local copy of the current rank) are assigned to different cores for asynchronous execution. Therefore, each rank needs to occupy a total of four Vector Core resources. These resources are automatically allocated by the system when the operator is dispatched and do not need to be allocated in advance.
- Notify: Each rank needs to synchronize with all other ranks. The synchronization operations include pre-synchronization and post-synchronization, which must be used in pairs. Therefore, each rank needs a total of six Notify resources. Each soft synchronization Notify resource is allocated 32 bytes of on-chip memory, and typically a space of about 1 KB to 1 MB can be reserved directly.
- Channel: Each rank needs to communicate with all other ranks, so each rank needs to establish three communication channels in total.

## Sample Code

The following uses the AllGather communication operator as an example to demonstrate the resource creation code snippet on the host:

1. Allocate context memory to store resource information.

    ```c
    uint64_t size = Size of the context memory to be allocated;
    void *ctx = nullptr;
    char *tag = Tag used to store resources;
    CommEngine engine = COMM_ENGINE_CPU_TS; // Use host-side memory to cache resources.
    HcclResult ret = HcclEngineCtxGet(comm, tag, engine, &ctx, &size);
    if (ret != HCCL_SUCCESS) {
       // That is, the resource represented by the tag has not been created before.
       HcclEngineCtxCreate(comm, tag, engine, size, ctx); 
    } else {
       // Indicates that resources have been created previously, so ctx can be used directly.
    }
    ```

2. Allocate Notify resources.

    ```c
    char tag[] = "allgather";
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;                   // AIV communication engine
    void* aivTagBufPtr = nullptr;                                      // Address of the soft synchronization tag area
    HcclEngineCtxCreate(comm, tag, engine, AIV_TAG_BUFF_LEN, &aivTagBufPtr);  // Create Notify resources.
    aclrtMemset(aivTagBufPtr, AIV_TAG_BUFF_LEN, 0, AIV_TAG_BUFF_LEN);  // Clear the tag area.
    ```

3. Register the Notify soft synchronization memory with the communicator, and establish a communication channel between every two ranks.

    ```c
    HcclMemHandle memHandle;
    CommMem regMem{COMM_MEM_TYPE_DEVICE, aivTagBufPtr, AIV_TAG_BUFF_LEN};
    HcclCommMemReg(comm, tag, &regMem, &memHandle); // Register the soft synchronization memory with the communicator.
    
    HcclChannelDesc channelDesc;
    HcclChannelDescInit(&channelDesc, 1);
    channelDesc.remoteRank = destRank;
    channelDesc.channelProtocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    channelDesc.memHandles = &memHandle;           // Specify the soft synchronization memory for exchange during link setup.
    channelDesc.memHandleNum = 1;
    ChannelHandle channelHandle;
    HcclChannelAcquire(comm, engine, &channelDesc, 1, &channelHandle);
    ```

4. Allocate communication memory.

    ```c
    // Allocate and obtain the local communication memory.
    void *localAddr;
    uint64_t localSize;
    HcclGetHcclBuffer(comm, &localAddr, &localSize);
    // Obtain the remote communication memory for subsequent read and write operations.
    void *remoteAddr;
    uint64_t remoteSize;
    HcclChannelGetHcclBuffer(comm, channelHandle, &remoteAddr, &remoteSize);
    // Obtain the remote soft synchronization tag area memory for subsequent synchronization operations.
    uint32_t memNum;
    CommMem* remoteMems;
    char** memTags;
    HcclChannelGetRemoteMems(comm, channelHandle, &memNum, &remoteMems, &memTags);
    void *remoteTagBufAddr = remoteMems[0].addr;
    ```
