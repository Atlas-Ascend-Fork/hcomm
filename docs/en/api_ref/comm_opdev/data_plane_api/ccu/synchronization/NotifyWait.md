# NotifyWait

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:19:21.273Z pushedAt=2026-09-30T09:06:30.904Z -->

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

Blocks and waits for the synchronization signal sent by the remote die through a channel inside a CCU kernel. It is a consumer-side API. The hardware blocks at this point until the corresponding notify bit is set by the remote side, after which execution continues.

It is applicable to cross-die scenarios (cross-die within the same device, cross-device within the same node, and cross-node). The corresponding producer-side API is [NotifyRecord](NotifyRecord.md).

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
CcuResult NotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channel | Input | Communication channel handle, which is the channel resource obtained through the link establishment process. For the definition of the `ChannelHandle` type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). The die bound to channel must be the same as the current die of this kernel. |
| localNotifyIdx | Input | Local notify slot index, which is the number (0-based) negotiated by both ends of the channel. It must be exactly the same as the `remoteNotifyIdx` value of `NotifyRecord` on the producer side. |
| mask | Input | 16-bit event mask that specifies the bit to wait for. The default value is `1`. It must be the same as the `mask` of `NotifyRecord` on the producer side. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | No kernel is currently in the registration phase (the API is called outside the kernel registration phase). |

> [!NOTE] Note
> This API does not validate channel/`localNotifyIdx` at the call site. The consistency of the die bound to the channel is uniformly validated by `HcommCcuKernelRegister` (based on all channels used by this kernel). If they are inconsistent, `CCU_E_PARA` is returned.

## Constraints

- `NotifyWait` can be called inside a hardware loop body.
- All channels in this kernel must belong to the same die. This consistency is uniformly validated by `HcommCcuKernelRegister`, and `CCU_E_PARA` is returned on inconsistency.
- `localNotifyIdx` must use the same value as `remoteNotifyIdx` of `NotifyRecord` on the producer side. This is agreed upon by both communication parties at the protocol layer, and the framework does not perform automatic pairing. A single `ChannelHandle` holds a maximum of 8 Notify resources, so the value range of this parameter is 0 to 7.
- Before the call, a corresponding `NotifyRecord` must already exist at the preceding position on the producer side; otherwise, the call is blocked permanently, causing a hardware-level deadlock.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: die B waits for die A to complete the write operation before reading data.
// Inside the CCU kernel function body of die B.
CcuResult DieReceiverKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*, so first cast it to the user input parameter structure.
    // Wait for the signal from die A, bit0 of local notify slot 3.
    // Corresponds to NotifyRecord(channel, /*remoteNotifyIdx=*/3, 0x1) on the die A side.
    NotifyWait(params->channel, /*localNotifyIdx=*/3, 0x1);

    // After this, the data written by die A can be safely read.
    RemoteAddr remote;
    LocalAddr dst;
    Variable len;
    Event evt;
    Read(params->channel, dst, remote, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
