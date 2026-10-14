# HcclKernelLaunchCfg

<!-- md-trans-meta sourceCommit=80101cd3106bc42b522595efad5c56429d70fbeb translatedAt=2026-09-28T06:38:20.483Z pushedAt=2026-09-28T08:25:39.647Z -->

## Description

Describes the configuration information for running a kernel function on AI CPU, including configuration parameters such as the timeout period.

## Prototype

```c
typedef struct {
    CommAbiHeader header;
    uint64_t timeOut;
    uint8_t reserved[104];
} HcclKernelLaunchCfg;
```

## Parameters

- **header**: ABI header, which contains information such as the version. For the type definition, see [CommAbiHeader](../../comm_opdev/datatype_definition/CommAbiHeader.md).
- **timeOut**: timeout period for the task scheduler to wait for task execution, in seconds.
- **reserved**: reserved field, 104 bytes in length, for future extension.
