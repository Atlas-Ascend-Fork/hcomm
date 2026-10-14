# HcclGetCommAsyncError

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:41:25.741Z pushedAt=2026-10-12T06:12:35.579Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

When the device NIC communication link in the cluster is unstable or network congestion occurs, "error cqe" is printed in the device log. This error is called an "RDMA ERROR CQE" error.

In the current version, this API can only be used to query whether an "RDMA ERROR CQE" error exists in the communicator.

> [!NOTE] Note
> This API is a synchronous API. That is, after the API is called, you need to wait for the result to be returned.

## Function Prototype

```c
HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult *asyncError)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator to be queried for error information.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| asyncError | Output | - **0**: No error occurs in the communicator.<br>  - **21**: An "RDMA ERROR CQE" error occurs in the communicator. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and the corresponding error code on failure.

## Constraints

- This API can be called only after a communicator is established.
- This API cannot be called after the communicator is destroyed.
- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

```c
// Initialize device resources.
aclInit(NULL);
aclrtSetDevice(devId);

// Create the communicator.
HcclComm hcclComm;
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);
HcclCommInitRootInfo(8, &rootInfo, 0, &hcclComm);

// Query whether an asynchronous error occurs in the communicator.
HcclResult asyncError;
HcclGetCommAsyncError(hcclComm, &asyncError);
if (asyncError == HCCL_E_REMOTE) {
    // The "RDMA ERROR CQE" error occurred in the communicator. Handle it accordingly.
}

// Destroy the communicator.
HcclCommDestroy(hcclComm);
aclFinalize();
```
