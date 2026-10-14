# HcclChannelGetHcclBuffer

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:46:35.928Z pushedAt=2026-10-12T06:12:35.591Z -->

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

Obtains the HCCL communication memory of the peer end of a specified channel.

## Function Prototype

```c
HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void **buffer, uint64_t *size)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| channel | Input | Communication channel handle.<br>For details about the ChannelHandle type, see [ChannelHandle](../../datatype_definition/ChannelHandle.md). |
| buffer | Output | Address of the HCCL communication memory. |
| size | Output | Size of the HCCL communication memory. The memory size is twice the value configured during communicator initialization or the value configured by the **HCCL_BUFFSIZE** environment variable, and defaults to 400 MB. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
// Set the device for the current thread.
aclrtSetDevice(0);

// Initialize the channel descriptor.
uint32_t channelNum = 1;
std::vector<HcclChannelDesc> channelDesc(channelNum);
HcclChannelDescInit(channelDesc.data(), channelNum);

// The peer end is Rank1.
channelDesc[0].remoteRank = 1;
channelDesc[0].channelProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
channelDesc[0].notifyNum = 3;

// Communicator handle.
HcclComm comm;
// Create the channel resource.
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
std::vector<ChannelHandle> channels(channelNum);
HcclChannelAcquire(comm, engine, channelDesc.data(), channelNum, channels.data());

// Obtain the address and size of the HCCL Buffer of the peer end (Rank1).
void *remoteBufferAddr;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channels[0], &remoteBufferAddr, &remoteBufferSize);
```
