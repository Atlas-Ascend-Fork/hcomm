# HcommChannelStatus

## 功能说明

通道建链状态枚举，用于解析[HcommChannelGetStatus](../control_plane_api/basic_resource_mgmt/HcommChannelGetStatus.md)接口出参statusList中每个通道的建链状态值。

## 类型定义

```c
typedef enum {
    HCOMM_CHANNEL_STATUS_READY = 0,
    HCOMM_CHANNEL_STATUS_CONNECTING = 1,
    HCOMM_CHANNEL_STATUS_FAILED_INTERNAL = 2,
} HcommChannelStatus;
```

## 枚举值说明

| 枚举值 | 值 | 描述 |
| --- | --- | --- |
| HCOMM_CHANNEL_STATUS_READY | 0 | 通道链路已就绪。整批通道就绪后方可进行通信操作。 |
| HCOMM_CHANNEL_STATUS_CONNECTING | 1 | 建链进行中，需继续调用[HcommChannelGetStatus](../control_plane_api/basic_resource_mgmt/HcommChannelGetStatus.md)接口推动建链。 |
| HCOMM_CHANNEL_STATUS_FAILED_INTERNAL | 2 | 建链失败。 |

## 使用说明

- 通道通过[HcommChannelCreate](../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md)或[HcommChannelCreateWithConfig](../control_plane_api/basic_resource_mgmt/HcommChannelCreateWithConfig.md)创建后不会立即完成建链，调用方需循环调用[HcommChannelGetStatus](../control_plane_api/basic_resource_mgmt/HcommChannelGetStatus.md)，依据本枚举判断建链进度，整批通道就绪后方可进行通信操作。
- 状态为HCOMM_CHANNEL_STATUS_CONNECTING时表示建链仍在进行，需继续调用接口推动建链；状态为HCOMM_CHANNEL_STATUS_READY时表示通道链路已就绪。
- 判定建链失败时，状态值大于等于HCOMM_CHANNEL_STATUS_FAILED_INTERNAL均视为建链失败。除上述枚举值外，当前版本还可能返回建链超时、本端资源不足、对端资源不足等细分状态值，仅供调用方辅助定位失败原因，不应作为判定依据。
