# HcclStuckDetectGetCounterAddr

<!-- md-trans-meta sourceCommit=999ba56f185da6fca7c6c072d4328220e4c8698a translatedAt=2026-11-13T09:40:00.000Z pushedAt=2026-11-13T09:40:00.000Z -->

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

This API is called on the host to obtain the device addresses of the stuck-detection counters for a communicator. The host only obtains and passes the addresses and must not dereference these device addresses directly. The device-side kernel reads the counters through these addresses.

Counter semantics:

- `headOpCounterAddr`: Head counter address; incremented on the device before an operator starts.
- `tailOpCounterAddr`: Tail counter address; incremented on the device after an operator completes.
- `head - tail` indicates the number of in-flight operators.
- `addOneMemAddr`: Address of the constant 1 on the device; the kernel uses it as the source of an atomic add or LocalReduce to increment head/tail.

Typical usage:

1. Create and initialize a V2 communicator on the host and ensure `stuck_detection` is enabled.
2. Call this API on the host to obtain `headOpCounterAddr`, `tailOpCounterAddr`, and `addOneMemAddr`.
3. Write these three values into the kernel arguments, together with the counter size `counterMemSize = sizeof(int32_t)` and the enable flag `isEnableCounter`.
4. The device-side kernel increments head/tail through the addresses and uses `head - tail` to determine whether the communicator is still making progress.

The following table shows the typical mapping to kernel arguments in the hccl AIV/CCU/AICPU integration:

| API Output | Typical Kernel Argument | Device-Side Usage |
| --- | --- | --- |
| `headOpCounterAddr` | `headCountMem` | Head counter address; incremented before an operator starts and read by the detector |
| `tailOpCounterAddr` | `tailCountMem` | Tail counter address; incremented after an operator completes and read by the detector |
| `addOneMemAddr` | `addOneMem` | Address of constant 1, used as the source of an atomic add or LocalReduce |

An address of 0 indicates that detection is disabled or address retrieval failed; the kernel should skip counter operations in this case. On the host, cache the result per communicator to avoid querying it on every kernel launch, and obtain it again after the communicator is destroyed or resumed. The detection threshold should be greater than normal queuing and NotifyWait latency; otherwise, normal waiting may be misreported.

## Function Prototype

```c
HcclResult HcclStuckDetectGetCounterAddr(
    HcclComm comm, uint64_t *headOpCounterAddr, uint64_t *tailOpCounterAddr, uint64_t *addOneMemAddr)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle. Only V2 communicators are supported.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| headOpCounterAddr | Output | Device address of the head counter; 0 when detection is disabled or address retrieval fails. |
| tailOpCounterAddr | Output | Device address of the tail counter; 0 when detection is disabled or address retrieval fails. |
| addOneMemAddr | Output | Device address of the constant 1 used by counter reduction; 0 when detection is disabled or address retrieval fails. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): Returns `HCCL_SUCCESS` on success and an error code on failure.

- `HCCL_E_PTR`: `comm` or any output pointer is null, or the internal communicator/manager is null.
- `HCCL_E_NOT_SUPPORT`: The communicator is a legacy communicator.
- `HCCL_E_INTERNAL`: Invalid communicator information, invalid counter address, or an exception occurred while querying the counter address.
- `HCCL_E_MEMORY`: Counter pool allocation failed or slots are exhausted.
- When detection is disabled, returns `HCCL_SUCCESS` with all three outputs set to 0.

## Constraints

- Only V2 communicators are supported; legacy communicators return `HCCL_E_NOT_SUPPORT`.
- This API can be called only on the host. The returned addresses are device virtual addresses and must not be dereferenced by the host; pass them to the device through kernel arguments.
- It is recommended to place the three outputs in the same kernel argument structure. On the device, read head/tail as `int32_t`; pass `addOneMemAddr` only as the address of constant 1.
- Even if a non-zero address is returned, do not use normal host memory access to read the counters; the counters are accessed only by the device-side kernel or CCU.
- Stuck detection is controlled by `stuck_detection:on/off` in `HCCL_DFS_CONFIG` and is enabled by default. The value is case-sensitive.
- When `cluster_heartbeat` is disabled, heartbeat links are not established and events are not broadcast to peers; however, the `ClusterMonitor` background thread may still start or be reused, and this API may still return valid addresses and participate in local detection.
- Stuck detection relies on the `ClusterMonitor` background thread. A communicator with `rankSize == 1` is not registered with `ClusterMonitor`; a pure single-rank process does not start the detection thread, so its counters are not sampled. If another multi-rank communicator on the same device has already started the thread, counters allocated on demand for the single-rank communicator may be sampled by that thread, but this is not guaranteed. Do not assume detection is active merely because this API returns valid addresses.
- The returned addresses are valid only while `comm` is alive and must not be used after the communicator is destroyed. A released slot may be reused by another communicator.
- Do not call this API concurrently with `HcclCommDestroy`.

## Example

### Host-Side Retrieval and Passing

The host obtains the addresses and passes them as kernel arguments:

```c
// Kernel arguments: filled by the host and used by the device-side kernel
typedef struct {
    uint64_t headCountMem;
    uint64_t tailCountMem;
    uint64_t addOneMem;
    uint32_t counterMemSize;
    uint32_t isEnableCounter;
} StuckDetectKernelArgs;

// Precondition: comm has been initialized as a V2 communicator by the communicator creation API
HcclComm comm = nullptr; // Placeholder: replace with an initialized V2 communicator handle
if (comm == nullptr) {
    return HCCL_E_PTR;
}
uint64_t headOpCounterAddr = 0;
uint64_t tailOpCounterAddr = 0;
uint64_t addOneMemAddr = 0;

HcclResult ret = HcclStuckDetectGetCounterAddr(
    comm, &headOpCounterAddr, &tailOpCounterAddr, &addOneMemAddr);
if (ret != HCCL_SUCCESS) {
    // Error handling
    return ret;
}

StuckDetectKernelArgs args = {};
args.headCountMem = headOpCounterAddr;
args.tailCountMem = tailOpCounterAddr;
args.addOneMem = addOneMemAddr;
args.counterMemSize = sizeof(int32_t);
args.isEnableCounter =
    (headOpCounterAddr != 0 && tailOpCounterAddr != 0 && addOneMemAddr != 0) ? 1 : 0;

// Pass args as kernel arguments; do not dereference these addresses on the host
```

### Device-Side Usage

The device-side kernel reads the counters (pseudocode):

```c
// args are the kernel arguments passed by the host
if (args->isEnableCounter != 0) {
    volatile int32_t *head = (volatile int32_t *)(uintptr_t)args->headCountMem;
    volatile int32_t *tail = (volatile int32_t *)(uintptr_t)args->tailCountMem;
    int32_t headValue = *head;
    int32_t tailValue = *tail;
    // Use head - tail to determine whether the communicator is still making progress
}

// args->addOneMem points to constant 1 and is used as the source for incrementing head/tail
```
