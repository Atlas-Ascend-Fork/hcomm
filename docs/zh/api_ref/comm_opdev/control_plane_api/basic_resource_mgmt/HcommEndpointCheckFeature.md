# HcommEndpointCheckFeature

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

该接口用于查询指定网络端点是否支持某种特性，无需实际创建端点。

通过传入端点描述信息和特性类型，查询底层硬件/网络驱动是否支持该特性，结果通过value参数返回。该接口主要用于在创建端点之前进行能力探测，以便上层根据硬件能力选择不同的通信策略。

当前支持查询NDA（NPU Direct RDMA Async）和Inline-Reduce（主机网卡内联规约）两种特性。

## 函数原型

```c
HcommResult HcommEndpointCheckFeature(HcommEndpointFeatureType featureType, const EndpointDesc *endpointDesc, bool *value);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| featureType | 输入 | 待查询的特性类型。<br>HcommEndpointFeatureType类型的定义请参见[HcommEndpointFeatureType](../../datatype_definition/HcommEndpointFeatureType.md)。 |
| endpointDesc | 输入 | 端点描述信息，用于指定待查询的端点属性（协议、地址、位置等），无需实际创建端点即可查询。<br>EndpointDesc类型的定义请参见[EndpointDesc](../../datatype_definition/EndpointDesc.md)。<br>该参数不能为空指针。 |
| value | 输出 | 特性支持情况的结果。<br>true表示支持该特性，false表示不支持。接口返回非0时value无实际含义。<br>该参数不能为空指针。 |

## 返回值

HcommResult：接口成功返回0（HCCL_SUCCESS）；返回5（HCCL_E_NOT_SUPPORT）仅表示设备不支持本接口，此时不返回查询结果（value无实际含义）；featureType传入不支持的值时返回0（HCCL_SUCCESS）且value为false；其他失败返回对应错误码。

## 约束说明

- 查询HCOMM_ENDPOINT_FEATURE_NDA与HCOMM_ENDPOINT_FEATURE_INLINE_REDUCE特性时，仅支持Ascend 950系列及后续代际产品，其他产品型号调用返回5（HCCL_E_NOT_SUPPORT）。
- 调用前无需创建Endpoint，仅需填充EndpointDesc描述信息。
- 查询HCOMM_ENDPOINT_FEATURE_NDA特性时，EndpointDesc中的协议需为COMM_PROTOCOL_ROCE，端点位置类型需为ENDPOINT_LOC_TYPE_HOST，否则返回false。
- 查询HCOMM_ENDPOINT_FEATURE_INLINE_REDUCE特性时，EndpointDesc中的协议需为COMM_PROTOCOL_ROCE，端点位置类型需为ENDPOINT_LOC_TYPE_HOST，否则返回false。
- featureType传入不支持的值时，返回0（HCCL_SUCCESS）且value为false，兼容调用方使用后续版本新增的特性枚举值。
- 返回5（HCCL_E_NOT_SUPPORT）仅表示设备不支持本接口，调用方应按“不支持该特性”处理并回退到其他通信方式，而非当作失败中断。
- 该接口主要用于能力探测阶段，不应在通信过程中频繁调用。

## 调用示例

```c
// 填充端点描述信息
EndpointDesc endpointDesc;
// 参考HcommEndpointCreate进行EndpointDesc的填充
...

// 查询是否支持NDA特性
bool isNdaSupported = false;
HcommResult ret = HcommEndpointCheckFeature(HCOMM_ENDPOINT_FEATURE_NDA, &endpointDesc, &isNdaSupported);
if (ret == HCCL_E_NOT_SUPPORT) {
    // 设备不支持本接口，value无实际含义，按不支持处理并回退
    isNdaSupported = false;
} else if (ret != HCCL_SUCCESS) {
    printf("Failed to check feature, ret = %d\n", ret);
    return ret;
}

if (isNdaSupported) {
    // 使用NDA特性进行通信
    // ...
} else {
    // 回退到其他通信方式
    // ...
}
```

```c
// 填充端点描述信息
EndpointDesc endpointDesc;
// 参考HcommEndpointCreate进行EndpointDesc的填充
...

// 查询是否支持Inline-Reduce特性
bool isInlineSupported = false;
HcommResult ret = HcommEndpointCheckFeature(HCOMM_ENDPOINT_FEATURE_INLINE_REDUCE, &endpointDesc, &isInlineSupported);
if (ret == HCCL_E_NOT_SUPPORT) {
    // 设备不支持本接口，value无实际含义，按不支持处理并回退
    isInlineSupported = false;
} else if (ret != HCCL_SUCCESS) {
    printf("Failed to check feature, ret = %d\n", ret);
    return ret;
}

if (isInlineSupported) {
    // 使用Inline-Reduce特性进行读写
    // ...
} else {
    // 回退到其他通信方式
    // ...
}
```
