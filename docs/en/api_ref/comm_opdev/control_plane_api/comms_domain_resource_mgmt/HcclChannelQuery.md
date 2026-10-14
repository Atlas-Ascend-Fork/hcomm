# HcclChannelQuery

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:47:05.462Z pushedAt=2026-10-12T06:12:35.593Z -->

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

Before calling [HcclChannelAcquire](HcclChannelAcquire.md), checks whether the channel corresponding to the specified communication channel description (**channelDescs**) already exists. If it exists, the corresponding channel handle is returned and can be directly reused; if it does not exist, **0** (a null handle) is returned, indicating that a new channel needs to be created in the subsequent call to [HcclChannelAcquire](HcclChannelAcquire.md).

## Function Prototype

```c
HcclResult HcclChannelQuery(HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| engine | Input | Communication engine type.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| channelDescs | Input | List of communication channel descriptions, with a length of **channelNum**. It must be initialized using [HcclChannelDescInit](HcclChannelDescInit.md).<br>For the definition of the HcclChannelDesc type, see [HcclChannelDesc](../../datatype_definition/HcclChannelDesc.md). |
| channelNum | Input | Number of communication channels. The value range of **channelNum** is (0, 1024 * 1024]. |
| channels | Output | List of query result handles, with a length of **channelNum**. If a channel already exists, the corresponding handle is returned; otherwise, **0** (a null handle) is returned. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

1. **channelDescs** must be initialized using [HcclChannelDescInit](HcclChannelDescInit.md).

2. Channels whose local and remote endpoints have different loc types (for example, between hostNIC and deviceNIC) cannot be reused. The query result is always **0**, indicating that a new channel is created each time [HcclChannelAcquire](HcclChannelAcquire.md) is called.

3. The channel handle returned by the query can be used in correspondence with the handle returned by [HcclChannelAcquire](HcclChannelAcquire.md).

4. Within the same communicator, this API must not be called concurrently with [HcclChannelAcquire](HcclChannelAcquire.md) or [HcclChannelDestroy](HcclChannelDestroy.md). The caller must ensure that the related calls are executed serially.

5. Within the same communicator, concurrent calls to this API are not supported. The caller must ensure that the calls are executed serially.
6. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

The following example queries communication channels in batches. The steps for constructing **channelDesc** and filling its fields are the same as those in the [HcclChannelAcquire](HcclChannelAcquire.md) example:

```c
// 1. Construct the channelDesc list.
HcclComm comm;   // Communicator that has been created.
CommEngine engine = CommEngine::COMM_ENGINE_CCU;
uint32_t channelNum = 2;
std::vector<HcclChannelDesc> channelDescVec(channelNum);
for (uint32_t idx = 0; idx < channelNum; idx++) {
    HcclChannelDesc channelDesc;
    CHK_RET(HcclChannelDescInit(&channelDesc, 1));
    // Fill the channelDesc fields as described in the [HcclChannelAcquire](HcclChannelAcquire.md) example.
    channelDescVec[idx] = channelDesc;
}

// 2. Query which channels already exist.
std::vector<ChannelHandle> channels(channelNum);
HcclResult ret = HcclChannelQuery(comm, engine, channelDescVec.data(), channelNum, channels.data());
if (ret != HCCL_SUCCESS) {
    // Error handling.
    return;
}
for (uint32_t idx = 0; idx < channelNum; idx++) {
    if (channels[idx] != 0) {
        // The channel already exists and can be reused directly.
    } else {
        // The channel does not exist. It will be created when HcclChannelAcquire is subsequently called.
    }
}
```
