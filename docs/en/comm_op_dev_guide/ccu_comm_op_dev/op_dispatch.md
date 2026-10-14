# Dispatching the Operator

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-24T07:08:02.721Z pushedAt=2026-09-28T05:59:29.461Z -->

After completing the communication operator kernel registration, developers need to dispatch the kernel function to a specific communication engine for execution.

For the CCU communication engine, developers need to call the **HcommCcuKernelLaunch** API to dispatch the kernel.

## Sample Code

Take the custom AllGather operator as an example. When using the CCU communication engine, the code snippet for dispatching the CCU kernel function on the host is as follows:

```c
uint64_t currentRankSliceInputOffset = 0; // Input address offset between devices
uint64_t currentRankSliceOutputOffset = sliceSize * myRank; // Destination address offset between devices
std::vector<uint64_t> taskArgs = {
    inputAddr,
    outputAddr,
    token,
    currentRankSliceInputOffset,
    currentRankSliceOutputOffset,
    sliceSize
};
CcuResult launchRet = HcommCcuKernelLaunch(thread, ccuKernel, taskArgs.data(), taskArgs.size()); // Dispatch the CCU task.
if (launchRet != CCU_SUCCESS) {
    HCCL_ERROR("[CcuTempAllGatherMesh1DMem2Mem::ExecOp] kernel launch failed, ccuRet -> %d", launchRet);
    return ConvertCcuToHccl(launchRet);
}
```
