# HcclRootInfo

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-09-28T06:41:43.261Z pushedAt=2026-09-28T08:37:18.845Z -->

## Description

Rank information of the root node, mainly including the host IP address and host port of the root node, as well as the unique identifier of the root node (obtained by concatenating information such as the device ID and timestamp).

## Prototype

```c
const uint32_t HCCL_ROOT_INFO_BYTES =  4108; // 4108: root info length
typedef struct HcclRootInfoDef {
    char internal[HCCL_ROOT_INFO_BYTES];
} HcclRootInfo;
```
