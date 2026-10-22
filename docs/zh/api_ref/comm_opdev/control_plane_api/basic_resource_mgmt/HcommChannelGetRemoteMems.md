# HcommChannelGetRemoteMems

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas推理系列产品：不支持
<!-- end id5 -->

## 功能说明

获取通信通道中交换得到的远端内存信息。

调用本接口前，应调用[HcommChannelCreate](HcommChannelCreate.md)创建通信通道，并重复调用[HcommChannelGetStatus](HcommChannelGetStatus.md)，直至通道进入就绪状态。

## 函数原型

```c
HcommResult HcommChannelGetRemoteMems(ChannelHandle channelHandle, uint32_t* memNum, CommMem** remoteMem, char*** memInfos);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| channelHandle | 输入 | 通道句柄。<br>ChannelHandle类型的定义请参见[ChannelHandle](../../datatype_definition/ChannelHandle.md)。 |
| memNum | 输出 | 远端内存的数量。 |
| remoteMem | 输出 | 远端内存数组。数组中的地址和长度来自建链时交换的远端内存信息。<br>CommMem类型的定义请参见[CommMem](../../datatype_definition/CommMem.md)。 |
| memInfos | 输出 | 远端内存标签字符串数组。 |

## 返回值

[HcommResult](../../datatype_definition/HcommResult.md)：接口成功返回0，其他失败。

## 约束说明

- remoteMem和memInfos指向通道内部缓存，调用者不得释放或修改。

## 调用示例

```c
EndpointHandle endpointHandle = nullptr;
// ... 创建端点的代码（省略）

// 创建多个通道
const uint32_t channelNum = 1;
HcommChannelDesc channelDescs[channelNum] = {0};
ChannelHandle channels[channelNum] = {0};

// ... 填充channelDescs（省略）

CommEngine engine = COMM_ENGINE_AIV;
HcommResult ret = HcommChannelCreate(endpointHandle, engine,
                                     channelDescs, channelNum, channels);
if (ret != 0) {
    printf("Failed to create channels, ret = %d\n", ret);
    HcommEndpointDestroy(endpointHandle);
    return ret;
}

// ... 等待通道就绪（省略）

uint32_t memNum = 0;
CommMem* remoteMem = NULL;
char** memInfos = NULL;

ret = HcommChannelGetRemoteMems(channels[0], &memNum, &remoteMem, &memInfos);
if (ret != 0) {
    return ret;
}
```
