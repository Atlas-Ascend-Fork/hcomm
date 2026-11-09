# HcommChannelGetStatus

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:55:27.614Z pushedAt=2026-10-08T08:28:44.300Z -->

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
| statusList | Output | Array of channel statuses, used to return the current status of each channel, corresponding one-to-one with **channelList**.<br>This parameter cannot be a null pointer.<br>An array allocated by the caller, with space for at least **listNum** elements.<br>Status values are defined as follows:<br>**0**: link establishment is complete and the channel is ready.<br>**1**: link establishment is in progress. Call this API again to advance the link establishment.<br>**2**: link establishment failure.<br>**3**: link establishment timeout.<br>**4**: link establishment failure due to insufficient local resources. In the current version, only CCU resource insufficiency can be identified and reported.<br>**5**: link establishment failure due to insufficient peer resources. In the current version, only CCU resource insufficiency can be identified and reported. |

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

std::vector<int32_t> statuses(channelNum, 0);
bool hasFailed = false;
while (true) {
    HcommResult ret = HcommChannelGetStatus(channels.data(), channelNum, statuses.data());
    if (ret != HCCL_SUCCESS) {
        // API call failed.
        break;
    }

    bool allReady = true;
    for (uint32_t i = 0; i < channelNum; i++) {
        if (statuses[i] >= 2 && statuses[i] <= 5) {
            // Link establishment failure or timeout. Exit.
            hasFailed = true;
            allReady = false;
            break;
        }
        if (statuses[i] != 0) {
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
