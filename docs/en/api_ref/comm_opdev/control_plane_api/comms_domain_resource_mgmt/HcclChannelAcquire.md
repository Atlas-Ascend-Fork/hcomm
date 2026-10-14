# HcclChannelAcquire

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:44:53.885Z pushedAt=2026-10-12T06:12:35.588Z -->

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

Obtains multiple communication channels based on a communicator. If the corresponding communication channels do not exist in the communicator, they are created directly.

Whether a channel is reused is determined by the unique channel identifier consisting of **commId** + **engine** + **remoterank** + **channelProtocol**.

## Function Prototype

```c
HcclResult HcclChannelAcquire(HcclComm comm, CommEngine engine, const HcclChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *channels)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| engine | Input | Communication engine type.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| channelDescs | Input | Communication channel description list. The list length is **channelNum**.<br>For the definition of the HcclChannelDesc type, see [HcclChannelDesc](../../datatype_definition/HcclChannelDesc.md). The list is obtained by calling [HcclRankGraphGetLinks](../topo_info_query/HcclRankGraphGetLinks.md) to obtain link information and then filling it in. |
| channelNum | Input | Number of communication channels. The value range of **channelNum** is (0, 1024 * 1024]. |
| channels | Output | Communication channel handle list. The length of the communication channel handle list is **channelNum**. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

1. When **CommEngine** is set to **CCU**, external configuration of **NotifyNum** is not supported, and four CCU Notify resources are allocated by default.

2. When **CommEngine** is set to **CCU**, exchanging additional custom memory is not supported. Only the **HcclBuffer** of the communicator can be exchanged.

3. In the AIV direct-drive URMA scenario, the number of memory regions participating in the exchange on the local and remote ends must be consistent; otherwise, channel creation fails.

4. Within the same communicator, this API does not support concurrent calls. The caller must ensure that calls are executed serially.

5. The communication protocols supported by each **CommEngine** depend on the chip model, as described below:

   <!-- npu="950" id6 -->
   For the Ascend 950PR&950DT products, the communication protocols supported by each communication engine are as follows:

   - COMM_ENGINE_CPU
     - COMM_PROTOCOL_ROCE
   - COMM_ENGINE_AICPU_TS
     - COMM_PROTOCOL_UBOE
     - COMM_PROTOCOL_UB_CTP
     - COMM_PROTOCOL_UB_RTP
     - COMM_PROTOCOL_ROCE
   - COMM_ENGINE_AIV
     - COMM_PROTOCOL_UB_CTP
     - COMM_PROTOCOL_UB_RTP
     - COMM_PROTOCOL_UB_MEM
     - COMM_PROTOCOL_ROCE
   - COMM_ENGINE_CCU (COMM_PROTOCOL_UB_RTP not supported)
     - COMM_PROTOCOL_UB_CTP
   <!-- end id6 -->

   <!-- npu="A3" id7 -->
   For the Atlas A3 products, the communication protocols supported by each communication engine are as follows:

   - COMM_ENGINE_AICPU_TS
     - COMM_PROTOCOL_ROCE
     - COMM_PROTOCOL_HCCS
     - COMM_PROTOCOL_HCCS_ONLY
   <!-- end id7 -->

   <!-- npu="910b" id8 -->
   For Atlas A2 products, the communication protocols supported by each communication engine are as follows:

   - COMM_ENGINE_CPU_TS
     - COMM_PROTOCOL_ROCE
     - COMM_PROTOCOL_HCCS
     - COMM_PROTOCOL_HCCS_ONLY
   <!-- end id8 -->
6. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

Take batch communication channels as an example:

```c
// 1. Call HcclRankGraphGetLinks to obtain link information.
CommLink *linkList = nullptr;
uint32_t listSize;
CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));

// 2. Traverse each CommLink and fill in HcclChannelDesc.
uint32_t channelNum = listSize;
std::vector<HcclChannelDesc> channelDescVec(channelNum);
for (uint32_t idx = 0; idx < listSize; idx++) {
  HcclChannelDesc channelDesc;
  HcclChannelDescInit(&channelDesc, 1);
  channelDesc.remoteRank = rank;

  CommLink link = linkList[idx];

  //  Core mapping: extract endpoint information from CommLink.
  channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
  channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
  channelDesc.localEndpoint.loc    = link.srcEndpointDesc.loc;
  channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
  channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
  channelDesc.remoteEndpoint.loc   = link.dstEndpointDesc.loc;
  channelDesc.channelProtocol     = link.linkAttr.linkProtocol;
  channelDesc.notifyNum = 4; // Specify the number of Notify as needed by the user.

  channelDescVec[idx] = channelDesc;
}

// 3. Create channels in batches.
HcclComm comm;
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
std::vector<ChannelHandle> channels(channelNum);
HcclChannelAcquire(comm, engine, channelDescVec.data(), channelNum, channels.data());
```
