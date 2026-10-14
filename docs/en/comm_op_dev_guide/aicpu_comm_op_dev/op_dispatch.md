# Dispatching the Operator

<!-- md-trans-meta sourceCommit=4bce6591eb0a4898412343ee53a224437de00bfe translatedAt=2026-08-11T07:05:08.981Z pushedAt=2026-08-20T11:39:14.560Z -->

After completing communication operator task orchestration, developers need to dispatch kernel functions to a specific communication engine for execution.

For the AI CPU communication engine, developers must first declare an operator information library file in JSON format. The file content is as follows:

```c
{
  "CustomAicpuKernel": {
    "opInfo": {
      "opKernelLib": "AICPUKernel",                // Fixed value
      "kernelSo": "<aicpu_kernel_so_name>",        // Dynamic link library on the AI CPU side, user-defined, for example: libp2p_aicpu_kernel.so
      "functionName": "<aicpu_kernel_func_name>"   // Name of the AI CPU kernel function, user-defined, for example: HcclLaunchP2PAicpuKernel
    }
  }
}
```

Define the AI CPU kernel function in the following format, where <aicpu_kernel_func_name\> is the name of the AI CPU kernel function. The function name must be consistent with the **functionName** field in the operator information library file (JSON file).

```c
extern "C" unsigned int <aicpu_kernel_func_name>(void *param)
{
    // Kernel implementation
}
```

## Sample Code

Taking the custom Send/Receive operator as an example. When using the AI CPU communication engine, the code snippet for dispatching the AI CPU kernel function on the host side is as follows:

```c
// The host stream notifies the device primary thread.
aclrtRecordNotify(g_notifies[0], stream);

std::string kernelName = "HcclLaunchP2PAicpuKernel";
aclrtFuncHandle funcHandle;
aclrtArgsHandle argsHandle;
ACLCHECK(aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle));
ACLCHECK(aclrtKernelArgsInit(funcHandle, &argsHandle));
aclrtParamHandle paraHandle;
aclrtKernelArgsAppend(argsHandle, &param, sizeof(OpParam), &paraHandle);
aclrtKernelArgsFinalize(argsHandle);

uint16_t NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;
aclrtLaunchKernelCfg cfg;
aclrtLaunchKernelAttr attr;
attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
cfg.numAttrs = 1;
cfg.attrs = &attr;
constexpr uint32_t blockDim = 1;

// Execute the algorithm orchestration on the device side.
aclrtLaunchKernelWithConfig(funcHandle, blockDim, stream, &cfg, argsHandle, nullptr);

// The host stream waits for notification from the device.
aclrtWaitAndResetNotify(g_notifies[1], stream, CUSTOM_TIMEOUT);
```

On the AI CPU side, you need to define the kernel function entry, which must be compiled to the device side:

```c
typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct AlgResourceCtx {
    ThreadHandle threadHandle;                              // Communication thread handle
    CommBuffer localBuffer;                                 // Local communication buffer
    CommBuffer remoteBuffer;                                // Remote communication buffer
    ChannelHandle channelHandle;                            // Communication channel resource
    uint32_t notifyIds[AICPU_CONTROL_NOTIFY_NUM];           // Device-side control Notify in AI CPU mode
};

struct OpParam {
    char tag[TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];             // Communicator name
    void* inputPtr = nullptr;                               // Operator input data address
    void* outputPtr = nullptr;                              // Operator output data address
    uint64_t count = 0;                                     // Data volume of the operator
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;        // Data type of the operator
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;     // Operator type
    AlgResourceCtx* resCtx = nullptr;                       // Resource context
};

// Kernel function executed on the AI CPU
extern "C" unsigned int HcclLaunchP2PAicpuKernel(OpParam *param)
{
    HCCL_INFO("Entry-%s, commName[%s], tag[%s]", __func__, param->commName, param->tag);
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) { 
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }

    // Obtain the primary thread on the device side
    ThreadHandle thread = param->resCtx->threadHandle;
    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed start batch mode");
        return 1;
    }

    // The primary thread waits for notification from the host stream.
    if (HcommAclrtNotifyWaitOnThread(thread, param->resCtx->notifyIds[0], CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to wait notify[%d] from host main stream", param->resCtx->notifyIds[0]);
        return 1;
    }

    // Execute task orchestration.
    if (ExecOp(*param, param->resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("orchestrate failed for op:%d", param->opType);
        return 1;
    }

    // The primary thread notifies the host stream.
    if (HcommAclrtNotifyRecordOnThread(thread, param->resCtx->notifyIds[1]) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to record host main stream");
        return 1;
    }

    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed end batch mode");
        return 1;
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s success, commName[%s], tag[%s]", __func__, param->commName, param->tag);
    return 0;
}
```
