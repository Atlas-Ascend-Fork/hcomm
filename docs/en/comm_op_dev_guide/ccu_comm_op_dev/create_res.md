# Creating Resources

<!-- md-trans-meta sourceCommit=b725e1d224bd966a2994c1cfb91c25e85d5fadf3 translatedAt=2026-09-24T07:07:21.733Z pushedAt=2026-09-28T05:59:29.459Z -->

## Communication Resource Computation

Communication operators depend on underlying hardware communication resources during execution. Therefore, before operator task scheduling and dispatch, the required communication resources must be allocated. HCCL communication resources are classified into four categories:

- **Thread**: Used to express concurrency within a single rank. Tasks on the same thread are executed in strict sequential order, while tasks on different threads can be executed concurrently. Regardless of the expansion mode, threads include one primary thread and several secondary threads. The primary thread controls the start and end of tasks on the secondary threads. The primary thread must be allocated, and the number of secondary threads to be allocated is calculated separately by each communication algorithm.
- **Notify**: Used to implement synchronization mechanisms, including inter-thread synchronization within a rank and inter-rank synchronization.
- **Channel**: Used for transferring data across ranks.
- **Kernel**: An algorithm executed on CCU hardware. Each kernel contains various CCU resources, including CCU instruction space, registers, on-chip cache, concurrent engines, and channel table entries. The number of resources and the CCU instruction content required by a kernel are translated and allocated when the kernel is registered using the **HcommCcuKernelRegister** API.

Different communication algorithms often require different numbers of communication resources. The following uses a single-server 4-device topology with the CCU communication engine as an example to explain the logic for calculating the number of communication resources required by the fully-connected Mesh communication algorithm.

- **Fully-connected mesh communication algorithm**

  The hardware topology of the Mesh algorithm is shown in the following figure. Each rank directly communicates with all other ranks.

  ![](figures/mesh.png)

  The number of communication resources required by the fully-connected Mesh communication algorithm in the preceding topology is as follows:

  - Thread: The Mesh algorithm of the CCU is integrated into one kernel, and the kernel needs to be dispatched on one thread. Therefore, one thread resource is required.
  - Notify: When the algorithm requires only one thread, no Notify resource on the thread is required.
  - Channel: Each rank needs to communicate with all other ranks. Therefore, each rank needs to establish three communication channels.
  - Kernel: The Mesh algorithm is implemented in one kernel. Therefore, one kernel resource is required.

In summary, for the fully-connected Mesh communication algorithm, the number of communication resources required by each rank is described in the following table.

**Table 1** Communication resources required by the algorithm in a single-server 4-device topology

| Communication Resource | Mesh Algorithm |
| --- | --- |
| Thread | 1 |
| Notify | 0 |
| Channel | 3 |
| Kernel | 1 |

## Sample Code

The following uses a custom AllGather communication operator as an example to illustrate its resource creation code snippet on the host.

Allocate thread resources:

```c
CommEngine engine = CommEngine::COMM_ENGINE_CCU;  // CCU communication engine
uint32_t threadNum = 1;             // Allocate one communication thread.
uint32_t notifyNumPerThread = 0;    // Do not allocate additional synchronization resources in the communication thread.
ThreadHandle thread;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);
```

Establish communication channels, with one channel established between the local rank and each peer rank:

```c
std::vector<ChannelHandle> kernelChannels;
kernelChannels.resize(rankSize - 1);
for(uint32_t remoteRank = 0; remoteRank < rankSize; remoteRank++) {
    if (remoteRank == myRank) {
        continue;
    }
    uint32_t netLayer = 0, listSize = 0;
    CommLink *linkList = nullptr;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize));  // Obtain the link information between srcRank and dstRank.
    HcclChannelDesc desc;
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
    bool protocolExists = false;
    for (uint32_t idx = 0; idx < listSize; idx++) {
        CommLink link = linkList[idx];
        if (link.linkAttr.linkProtocol == protocol) {
            desc.remoteRank = remoteRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
            protocolExists = true;
            break;
        }
    }
    CHK_RET(HcclChannelAcquire(comm, param.engine, &desc, 1, &kernelChannels[channelIndex])); // Obtain the channel handle.
    channelIndex++;
}
```

Register the kernel:

```c
// Set the kernel function name and function pointer.
char kernelFuncName[64];
strcpy_s(kernelFuncName, sizeof(kernelFuncName), "CcuAllGatherMesh1DMem2MemKernel");
kernelFunc = reinterpret_cast<void *>(CcuAllGatherMesh1DMem2MemKernel);
auto kernelArg = std::make_shared<CcuKernelArgAllGatherMesh1DMem2Mem>(rankSize, myRank);
// Bind the channel to the kernel.
auto* kernelArgBase = static_cast<CcuKernelArgBase*>(kernelArg.get());
for (uint32_t i = 0; i < kernelChannels.size(); ++i) {
    kernelArgBase->channels[i] = kernelChannels[i];  // Save the channel handle in kernelArgBase.
}
kernelArgBase->channelCount = static_cast<uint32_t>(kernelChannels.size());
// Obtain the insHandle.
CcuInsHandle insHandle{0};
uint32_t insNum = 0;
CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
CHK_PRT_RET(insNum != 1,
    HCCL_ERROR("[GetCcuKernel] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
    HCCL_E_INTERNAL);
// Register the kernel.
CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
if (regStartRet != CCU_SUCCESS) {
    HCCL_ERROR("ccu kernel register start failed: ccuRet -> %d", regStartRet);
    return ConvertCcuToHccl(regStartRet);
}
CcuKernelHandle kernelHandle;
constexpr uint32_t dieId = 0;
constexpr uint32_t kernelArgNum = 1;
const void *kernelArgsArr[] = { kernelArg.get() }; // Construct the input parameter pointer array based on the HcommCcuKernelRegister signature.
CcuResult regRet = HcommCcuKernelRegister(insHandle, dieId, kernelFuncName, reinterpret_cast<void*>(kernelFunc), kernelArgsArr, kernelArgNum, &kernelHandle); // Register the kernel.
if (regRet != CCU_SUCCESS) {
    HCCL_ERROR("ccu kernel register failed: ccuRet -> %d", regRet);
    return ConvertCcuToHccl(regRet);
}
CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
if (regEndRet != CCU_SUCCESS) {
    HCCL_ERROR("ccu kernel register end failed: ccuRet -> %d", regEndRet);
    return ConvertCcuToHccl(regEndRet);
}
```
