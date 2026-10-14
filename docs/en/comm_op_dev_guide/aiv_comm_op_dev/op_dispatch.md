# Dispatching the Operator

<!-- md-trans-meta sourceCommit=4bce6591eb0a4898412343ee53a224437de00bfe translatedAt=2026-08-11T07:10:27.637Z pushedAt=2026-08-20T11:39:14.566Z -->

After completing the scheduling of communication operator tasks, developers need to dispatch the kernel function to a specific communication engine for execution.

Define the AIV kernel function in the following format, where `<aiv_kernel_func_name>` is the name of the AIV kernel function.

```c
extern "C" __global__ __aicore__ void <aiv_kernel_func_name>(void *param)
{
    // Kernel implementation
}
```

## Sample Code

Taking the custom AllGather operator as an example, when using the AIV communication engine, the code snippet for dispatching the AIV kernel function on the host is as follows:

```c
// Obtain the number of control cores.
aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit);

// Obtain the handle from the kernel binary file. This can be loaded only once globally.
char realPath[] = "hccl_aiv_all_gather_op.o";
aclrtBinHandle binHandle;
aclrtBinaryLoadOptions loadOptions = {0};
aclrtBinaryLoadOption option;
loadOptions.numOpt = 1;
loadOptions.options = &option;
option.type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD;
option.value.cpuKernelMode = 1;
aclrtBinaryLoadFromFile(realPath, &loadOptions, &binHandle);

// Obtain the function handle of the corresponding kernel.
char funcName[] = "aiv_all_gather_bfloat16_t";
aclrtFuncHandle funcHandle;
aclrtBinaryGetFunction(binHandle, funcName, &funcHandle);

// Prepare parameters for kernel launch.
aclrtLaunchKernelCfg cfg;
aclrtLaunchKernelAttr attr[3];
attr[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
attr[0].value.schemMode = 1;
attr[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
attr[1].value.timeoutUs.timeoutLow = 1800 * 1000000;
attr[1].value.timeoutUs.timeoutHigh = 0;
attr[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
attr[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
cfg.numAttrs = 3;
cfg.attrs = attr;

// Launch the AIV kernel.
aclrtLaunchKernelWithHostArgs(funcHandle, numBlocksLimit, stream, &cfg, args, argsSize, nullptr, 0);
```

On the device, the AIV kernel function entry must be defined, and the function must be compiled and linked into a separate binary .o file:

```c
#define EXPORT_AIV_META_INFO(kernel_name) \
static const struct FunLevelKType kernel_name##_kernel_type_section __attribute__ \
((used, section (".ascend.meta." #kernel_name))) \
= {{F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}}

extern "C" __global__ __aicore__ void aiv_all_gather_bfloat16_t(GM_ADDR buffIn, ...) {
    return AivAllGatherMesh<bfloat16_t>();
}
EXPORT_AIV_META_INFO(aiv_all_gather_bfloat16_t)
```
