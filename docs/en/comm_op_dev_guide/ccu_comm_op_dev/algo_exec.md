# Executing Algorithms

<!-- md-trans-meta sourceCommit=bbc253bc4059ffbc8cc6b575bbf0ec1bf6cec5ed translatedAt=2026-09-24T07:06:58.034Z pushedAt=2026-09-28T05:59:29.458Z -->

After a kernel is dispatched, the CCU hardware executes algorithms based on the tasks scheduled in the kernel, completing data movement and synchronization.

## Algorithm Orchestration Procedure

The CCU kernel receives two types of parameters from the host: **KernelArg** and **TaskArg**.

- **KernelArg**: Contains parameters required for algorithm orchestration, including the rank ID, communicator size, and reduction type of reduction-type operators. These parameters are passed as input parameters to the kernel function.
- **TaskArg**: Contains parameters required for algorithm execution, including the input address, output address, and token. These parameters need to be passed through **HcommCcuKernelLaunch** and dynamically loaded using the Load function in the kernel function.

The algorithm in the kernel mainly includes the following steps:

1. Initialize resources.

    Variables such as **Variable**, **LocalAddr**, **RemoteAddr**, and **CompetedEvent** that the algorithm requires must be declared. If a **Variable** is to be written by a remote rank, it must be bound to a channel during initialization, for example:

    ```c
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);
    AllGatherMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    
    // Initialize resources.
    ctx.output.resize(ctx.arg->rankSize);
    ctx.token.resize(ctx.arg->rankSize);
    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < ctx.arg->rankSize; peerId++) {
        if (peerId != ctx.arg->rankId) {
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    ```

2. Load parameter values from **TaskArg** into CCU variables, typically including the input address, output address, and token. The loading is implemented by calling the **Load** API.

    ```c
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
    ```

3. Perform pre-synchronization to synchronize information such as the local input address to the peer rank and wait for synchronization from the peer rank.

    Commonly used data plane APIs are **NotifyRecord** and **NotifyWait**. During pre-synchronization, you can also write the value of a **Variable** to the peer end, which requires the **WriteVariableWithNotify** API, for example:

    ```c
    // Write the value in the output variable to the peer end and synchronize it to bit 1 of synchronization register 0.
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << OUTPUT_XN_ID));
    }
    ```

4. Perform data movement. Commonly used data plane APIs include **Write**, **Read**, and **LocalCopy**.
5. Perform post-synchronization to notify the peer rank that the data movement is complete and wait for synchronization from the peer rank.

## Sample Code

Taking the custom AllGather operator as an example, the task scheduling code snippet for the CCU engine is as follows:

```c
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;
CcuResult CcuAllGatherMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);
    AllGatherMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    // 1. Initialize resources.
    ctx.output.resize(ctx.arg->rankSize);
    ctx.token.resize(ctx.arg->rankSize);
    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < ctx.arg->rankSize; peerId++) {
        if (peerId != ctx.arg->rankId) {
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    // 2. Load parameters.
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));
    // 3. Perform pre-synchronization.
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit));
    }
    // 4. Execute algorithms.
    ccu::LocalAddr src;
    ccu::LocalAddr localDst;
    std::vector<ccu::RemoteAddr> dst;
    dst.resize(ctx.arg->rankSize);
    src.addr = ctx.input;
    src.addr += ctx.currentRankSliceInputOffset;
    src.token = ctx.token[ctx.arg->rankId];
    for (uint32_t rankIdx = 0; rankIdx < ctx.arg->rankSize; rankIdx++) {
        if (rankIdx == ctx.arg->rankId) {
            localDst.addr = ctx.output[rankIdx];
            localDst.addr += ctx.currentRankSliceOutputOffset;
            localDst.token = ctx.token[rankIdx];
        } else {
            dst[rankIdx].addr = ctx.output[rankIdx];
            dst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
            dst[rankIdx].token = ctx.token[rankIdx];
        }
    }
    CCU_IF(ctx.sliceSize != 0)
    {
        uint32_t channelId = 0;
        for (uint64_t rankIdx = 0; rankIdx < ctx.arg->rankSize; rankIdx++) {
            const uint16_t mask = 1 << rankIdx;
            if (rankIdx != ctx.arg->rankId) {
                CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelId], dst[rankIdx], src, ctx.sliceSize, ctx.event, mask)); // Write local device data to the remote address.
                channelId++;
            } else {
                CCU_CHK_RET(ccu::LocalCopy(localDst, src, ctx.sliceSize, ctx.event, mask)); // Copy the data of this rank to the local destination.
            }
        }
    }
    const uint16_t totalMask = (1 << ctx.arg->rankSize) - 1;
    CCU_CHK_RET(ccu::EventWait(ctx.event, totalMask)); // Wait for data movement of the local device to complete.
    // 5. Perform post-synchronization.
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID)); // Wait for data movement of the remote device to complete.
    }
    return CcuResult::CCU_SUCCESS;
}
```
