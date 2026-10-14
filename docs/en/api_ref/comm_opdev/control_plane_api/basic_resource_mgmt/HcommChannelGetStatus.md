# HcommChannelGetStatus

<!-- md-trans-meta sourceCommit=05dc726adbd2c72ee36c39dc5844045f412eb39d translatedAt=2026-10-08T11:43:56.238Z pushedAt=2026-10-12T06:12:35.586Z -->

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

Queries the link establishment status of communication channels and advances the link establishment state machine by one step. Each call performs one round of link establishment advancement for all channels passed in and returns the current status. The caller must call this API repeatedly until the channel status becomes ready or failed. Communication operations can be performed only after the channel is ready.


## Function Prototype

```c
HcommResult HcommChannelGetStatus(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channelList | Input | Array of channel handles whose statuses are to be queried. Each element identifies a created communication channel.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../datatype_definition/ChannelHandle.md).<br>This parameter cannot be a null pointer. Each channel handle in the array must be a valid handle created by [HcommChannelCreate](HcommChannelCreate.md). |
| listNum | Input | Number of channels to be queried.<br>Unit: number. Value range: [1, 1048576].<br>This parameter must be greater than 0. |
| **statusList** | Output | Channel status array, used to return the current status of each channel, corresponding one-to-one with **channelList**.<br>This parameter cannot be a null pointer.<br>An array allocated by the caller, with space for at least **listNum** elements.<br>The element type of this array is **int32_t**, and each element takes a value from the **HcommChannelStatus** enumeration values. For its definition and usage, see [HcommChannelStatus](../../datatype_definition/HcommChannelStatus.md). |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- The length of the **channelList** array must be consistent with the **listNum** parameter.
- **statusList\[i\]** corresponds to **channelList\[i\]** one by one, indicating the status of the *i*th channel.
- The same **ChannelHandle** cannot be accessed concurrently in multiple threads. That is, the link establishment status query, communication, and destruction operations of the same channel must be executed serially.

## Example

```c
uint32_t channelNum = 50;
std::vector<ChannelHandle> channels(channelNum);
// Refer to HcommChannelCreate for channel creation.
...

std::vector<int32_t> statuses(channelNum, HCOMM_CHANNEL_STATUS_READY);
bool hasFailed = false;
while (true) {
    HcommResult ret = HcommChannelGetStatus(channels.data(), channelNum, statuses.data());
    if (ret != HCCL_SUCCESS) {
        // API call failed.
        break;
    }

    bool allReady = true;
    for (uint32_t i = 0; i < channelNum; i++) {
        if (statuses[i] >= HCOMM_CHANNEL_STATUS_FAILED_INTERNAL) {
            // Link establishment fails (including sub-scenarios such as link establishment timeout and insufficient resources). Exit.
            hasFailed = true;
            allReady = false;
            break;
        }
        if (statuses[i] != HCOMM_CHANNEL_STATUS_READY) {
            allReady = false;
        }
    }
    if (allReady || hasFailed) {
        break;
    }
    // Link establishment in progress. Sleep for a while and retry.
    usleep(1000);
}

if (hasFailed) {
    // Link establishment failed or timed out. Perform error handling.
    // ...
    return;
}

 // Channel ready. Perform communication operations.
 // ...
```
