# HcclKernelFuncInfo

<!-- md-trans-meta sourceCommit=80101cd3106bc42b522595efad5c56429d70fbeb translatedAt=2026-09-28T06:37:59.553Z pushedAt=2026-09-28T08:24:32.264Z -->

## Description

Describes AI CPU kernel function information, including the dynamic library name, kernel function name, parameter address, and parameter size. This structure is used to specify the kernel function to be launched and its parameters.

## Prototype

```c
typedef struct {
    char kernelSoName[HCCL_KERNEL_SO_NAME_MAX_LEN];
    char kernelFuncName[HCCL_KERNEL_FUNC_NAME_MAX_LEN];
    void *args;
    uint32_t argSize;
} HcclKernelFuncInfo;
```

## Parameters

- **kernelSoName**: dynamic library name, used to specify the file name of the dynamic library that contains the kernel function. The maximum length is 256 bytes (**HCCL_KERNEL_SO_NAME_MAX_LEN**).
- **kernelFuncName**: kernel function name, used to specify the name of the kernel function to be called in the dynamic library. The maximum length is 256 bytes (**HCCL_KERNEL_FUNC_NAME_MAX_LEN**).
- **args**: kernel function parameter address, a pointer to the parameter data. When **argSize** is greater than **0**, this parameter cannot be a null pointer.
- **argSize**: size of the kernel function parameter, in bytes.

## Related Constants

```c
const uint32_t HCCL_KERNEL_SO_NAME_MAX_LEN = 256;    // Maximum length of the dynamic library name.
const uint32_t HCCL_KERNEL_FUNC_NAME_MAX_LEN = 256;  // Maximum length of the function name.
```
