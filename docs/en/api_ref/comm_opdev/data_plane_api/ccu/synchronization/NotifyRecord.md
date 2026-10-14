# NotifyRecord

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:19:16.067Z pushedAt=2026-09-30T09:05:09.329Z -->

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
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Sends a synchronization signal to a remote die through a channel within a CCU kernel. It is a producer-side API. At runtime, it sends a die-to-die signal packet through the channel (HCCS is used for cross-die communication within the same device, and RDMA is used for cross-device or cross-node communication) to set the notify bit of the corresponding remote die.

This API applies to cross-die scenarios (cross-die within the same device, cross-device within the same node, and cross-node). The corresponding consumer-side API is [NotifyWait](NotifyWait.md).

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
CcuResult NotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channel | Input | Communication channel handle, which is the channel resource obtained through the link establishment process. For the definition of the `ChannelHandle` type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). The die bound to the channel must be the same as the current die of this kernel. |
| remoteNotifyIdx | Input | Remote notify slot index, which is the number (0-based) negotiated and agreed upon by both ends of the channel. It corresponds to `localNotifyIdx` of the consumer-side `NotifyWait`, and the numbers on both sides must be exactly the same. |
| mask | Input | 16-bit event mask, specifying the bit to be set. The default value is `1`. It must be consistent with the `mask` of `NotifyWait` on the consumer side. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | No kernel is currently in the registration phase (the API is called outside the kernel registration phase). |

> [!NOTE] Note
> This API returns `CCU_SUCCESS` at the call site and does not verify the channel/`remoteNotifyIdx`. The consistency of the die bound to the channel is uniformly verified by `HcommCcuKernelRegister` (based on all channels used during this kernel), and `CCU_E_PARA` is returned if they are inconsistent. This API does not intercept calls inside the hardware loop body. Do not use it inside the loop body (see [Loop](../execution_control/Loop.md)).

## Constraints

- All channels in this kernel must belong to the same die. This consistency is uniformly validated by `HcommCcuKernelRegister`, and `CCU_E_PARA` is returned on inconsistency.
- `remoteNotifyIdx` and `localNotifyIdx` of `NotifyWait` on the consumer side must use the same value, which is agreed upon by both communication parties at the protocol layer. The framework does not perform automatic pairing. A single `ChannelHandle` holds a maximum of 8 Notify resources, so the value range of this parameter is 0 to 7.
- `NotifyRecord` is on the producer side. `NotifyWait` on the consumer side must arrive after `NotifyRecord` (in hardware, "Record before Wait", meaning production precedes consumption). If the consumer side cannot receive the corresponding `NotifyRecord` (not initiated, `localNotifyIdx`/`mask` mismatch, etc.), it will be permanently blocked, causing a hardware-level deadlock.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: After die A completes the write operation, it notifies die B that the data can be read.
// Inside the CCU kernel function body of die A.
CcuResult DieSenderKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter structure first.
    // ... Perform the write operation to write data to the peer memory ...

    // Notify die B that bit0 of notify slot 3 is complete.
    NotifyRecord(params->channel, /*remoteNotifyIdx=*/3, 0x1);
    return CCU_SUCCESS;
}

// Inside the CCU kernel function body of die B (corresponding to the kernel of die A)
CcuResult DieReceiverKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter structure first.
    // Wait for the signal from die A, bit0 of local notify slot 3.
    NotifyWait(params->channel, /*localNotifyIdx=*/3, 0x1);

    // After this, the data written by die A can be safely read.
    return CCU_SUCCESS;
}
```
