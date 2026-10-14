# HcclKernelLaunchCfgInit

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:26:56.814Z pushedAt=2026-09-29T01:48:27.448Z -->

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

Initializes the **HcclKernelLaunchCfg** structure to describe the configuration information for running a kernel function on the AI CPU.

This API first fills the entire structure with 0xFF, and then sets the ABI header information (including the version number, magic number, and structure size) to put the structure into a usable initial state. After calling this API, developers need to set fields such as **timeOut** as required by the actual task, and then pass the configuration to [HcclAicpuKernelLaunch](./HcclAicpuKernelLaunch.md) to dispatch the task.

## Function Prototype

```c
static inline HcclResult HcclKernelLaunchCfgInit(HcclKernelLaunchCfg *kernelLaunchCfg)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| kernelLaunchCfg | Output | KernelLaunch configuration parameters to be initialized.<br>For the definition of the HcclKernelLaunchCfg type, see [HcclKernelLaunchCfg](./data_type_definition/HcclKernelLaunchCfg.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and **HCCL_E_PTR** when a null pointer is passed in.

## Constraints

- This API only initializes the ABI header information. The remaining fields of the structure are filled with 0xFF. After initialization, the caller needs to set the service fields (such as **timeOut**) as required.
- The pointer passed in must point to an **HcclKernelLaunchCfg** object with allocated memory and cannot be a null pointer.

## Example

```c
// Initialize the kernel function configuration.
HcclKernelLaunchCfg kernelLaunchCfg;
HcclKernelLaunchCfgInit(&kernelLaunchCfg);
kernelLaunchCfg.timeOut = 60;  // Set the timeout, in seconds.

// Run the AI CPU kernel function.
HcclResult ret = HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, userStream, &kernelLaunchCfg);
if (ret != HCCL_SUCCESS) {
    // Handle the error.
}
```
