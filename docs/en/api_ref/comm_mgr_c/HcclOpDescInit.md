# HcclOpDescInit

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:27:09.357Z pushedAt=2026-09-29T01:51:18.213Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Initializes the **HcclOpDesc** structure to describe the operator information when the AI CPU runs a kernel function.

This API first fills the entire structure with 0xFF and then sets the ABI header information (including the version number, magic number, and structure size) to put the structure into a usable initial state. After calling this API, developers need to set fields such as **opDescType**, **opName**, and union members (for example, **p2p**) as required by the actual communication task, and then pass the structure to [HcclAicpuKernelLaunch](./HcclAicpuKernelLaunch.md) to dispatch the task.

## Function Prototype

```c
static inline HcclResult HcclOpDescInit(HcclOpDesc *opDesc)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| opDesc | Output | Operator description parameter to be initialized.<br>For the definition of the HcclOpDesc type, see [HcclOpDesc](./data_type_definition/HcclOpDesc.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and **HCCL_E_PTR** when a null pointer is passed in.

## Constraints

- This API only initializes the ABI header information. The remaining fields of the structure are filled with 0xFF. After initialization, callers need to set the service fields (such as **opDescType**, **p2p.buffer**, and **p2p.cmdType**) as required.
- The pointer passed in must point to an **HcclOpDesc** object with allocated memory and cannot be a null pointer.

## Example

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
aclrtStream userStream;
aclrtCreateStream(&userStream);
ThreadHandle aicpuThreadHandle;
HcclResult ret = HcclThreadAcquire(comm, engine, 1, 1, &aicpuThreadHandle);
if (ret != HCCL_SUCCESS) {
    // Handle the error.
}

// Initialize the operator description parameter.
HcclOpDesc opInfo;
HcclOpDescInit(&opInfo);
opInfo.opDescType = 1;  // Represents a Send or Receive operator.
opInfo.p2p.buffer = sendBuffer;
opInfo.p2p.cmdType = HCCL_CMD_SEND;
opInfo.p2p.dataType = HCCL_DATA_TYPE_FP32;
opInfo.p2p.count = dataSize;
opInfo.p2p.remoteRank = destRank;
opInfo.p2p.unfoldStream = userStream;

// Initialize the kernel function information.
HcclKernelFuncInfo funcInfo;
strncpy_s(funcInfo.kernelSoName, HCCL_KERNEL_SO_NAME_MAX_LEN, "libscatter_aicpu_kernel.so", strlen("libscatter_aicpu_kernel.so") + 1);
strncpy_s(funcInfo.kernelFuncName, HCCL_KERNEL_FUNC_NAME_MAX_LEN, "HcclLaunchP2pAicpuKernel", strlen("HcclLaunchP2pAicpuKernel") + 1);
funcInfo.args = kernelArgs;
funcInfo.argSize = argsSize;

// Initialize the kernel function configuration.
HcclKernelLaunchCfg kernelLaunchCfg;
HcclKernelLaunchCfgInit(&kernelLaunchCfg);
kernelLaunchCfg.timeOut = 60;

// Run the AICPU kernel function.
ret = HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, userStream, &kernelLaunchCfg);
if (ret != HCCL_SUCCESS) {
    // Handle the error.
}
```
