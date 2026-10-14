# HcclRankGraphGetLinks

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:55:35.599Z pushedAt=2026-10-12T06:12:35.615Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Given a communicator and a topology layer ID, queries the communication link information between the source rank and the destination rank.

Take Atlas A3 products as an example:

- Example 1: The source rank and destination rank are in two different SuperPoDs.

  netLayer = 0, no connection.

  netLayer = 1, no connection.

  netLayer = 2, RDMA connection.

- Example 2: The source rank and destination rank are in the same SuperPoD but not in the same AI server.

  netLayer = 0, no connection.

  netLayer = 1, HCCS connection.

  netLayer = 2, no connection.

- Example 3: The source rank and destination rank are in the same AI server but not in the same NPU.

  netLayer = 0, HCCS connection.

  netLayer = 1, no connection.

  netLayer = 2, no connection.

## Function Prototype

```c
HcclResult HcclRankGraphGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink **links, uint32_t *linkNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| **comm** | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| **netLayer** | Input | Topology layer ID. |
| **srcRank** | Input | Source rank ID. |
| **dstRank** | Input | Destination rank ID. |
| **links** | Output | Communication link list.<br>For the definition of the CommLink type, see [CommLink](../../datatype_definition/CommLink.md). |
| **linkNum** | Output | Number of communication links. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- The returned memory is managed by the library. The caller must not release it.
- Copy the returned data in a timely manner. Repeated calls in the same communicator may invalidate the previous result.
- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
HcclComm comm;
CommLink *links;
uint32_t linkNum;
uint32_t netlayer = 0;
// Query the link between rank0 and rank1 within the server.
HcclRankGraphGetLinks(comm, netlayer, 0, 1, &links, &linkNum);
```
