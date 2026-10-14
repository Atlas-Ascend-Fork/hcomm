# HcclChannelDestroy

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:46:19.179Z pushedAt=2026-10-12T06:12:35.591Z -->

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

Destroys the communication channels created in a specified communicator and releases the corresponding channel resources. After destruction, the channel handles can no longer be used. To use them again, call [HcclChannelAcquire](HcclChannelAcquire.md) to obtain new handles.

## Function Prototype

```c
HcclResult HcclChannelDestroy(HcclComm comm, const ChannelHandle* channels, uint32_t channelNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| channels | Input | List of communication channel handles to be destroyed. The list length is **channelNum**. The handles must be valid handles returned by [HcclChannelAcquire](HcclChannelAcquire.md). |
| channelNum | Input | Number of communication channels. The value range of **channelNum** is (0, 1024 * 1024]. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

1. This API currently supports destroying only the channels of the CCU communication engine.

2. The handles in channels must be valid channel handles obtained through [HcclChannelAcquire](HcclChannelAcquire.md). After destruction, these handles can no longer be used.

3. If this API is called again with an invalid handle or a destroyed handle, **HCCL_E_NOT_FOUND** is returned.

4. During batch destruction, if some channels fail to be destroyed, this API returns the error code of the first failure and still tries to destroy the remaining channels.

5. Within the same communicator, this API must not be called concurrently with [HcclChannelAcquire](HcclChannelAcquire.md) or [HcclChannelQuery](HcclChannelQuery.md). The caller must ensure that the related calls are executed serially.

6. Within the same communicator, this API does not support concurrent calls. The caller must ensure that the calls are executed serially.
7. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

The following example demonstrates acquiring, using, and destroying a channel. For the steps of channel acquisition (including **channelDesc** construction and field filling), see the calling example in [HcclChannelAcquire](HcclChannelAcquire.md):

```c
// 1. Obtain the communication channel.
HcclComm comm;   // Created communicator.
CommEngine engine = CommEngine::COMM_ENGINE_CCU;
uint32_t channelNum = 2;
std::vector<HcclChannelDesc> channelDescVec(channelNum);
for (uint32_t idx = 0; idx < channelNum; idx++) {
    HcclChannelDesc channelDesc;
    CHK_RET(HcclChannelDescInit(&channelDesc, 1));
    // Fill in the channelDesc fields as shown in the calling example in [HcclChannelAcquire](HcclChannelAcquire.md).
    channelDescVec[idx] = channelDesc;
}
std::vector<ChannelHandle> channels(channelNum);
HcclResult ret = HcclChannelAcquire(comm, engine, channelDescVec.data(), channelNum, channels.data());
if (ret != HCCL_SUCCESS) {
    // Error handling.
    return;
}

// 2. Use the channel to complete the communication operation.
// ...

// 3. Destroy the communication channel.
ret = HcclChannelDestroy(comm, channels.data(), channelNum);
if (ret != HCCL_SUCCESS) {
    // Error handling.
    return;
}
```
