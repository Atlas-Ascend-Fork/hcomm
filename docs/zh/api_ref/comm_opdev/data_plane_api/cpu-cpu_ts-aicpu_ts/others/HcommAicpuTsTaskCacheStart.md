# HcommAicpuTsTaskCacheStart

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

在cache miss场景下，算子展开前调用该接口，通知AICPU Task Cache开始缓存task。该接口会将传入的内存基址和大小信息保存到cache entry中，后续算子展开过程中下发的SQE/WQE将被捕获并缓存。

## 函数原型

```c
HcommResult HcommAicpuTsTaskCacheStart(const char *tag, void **addrs, uint64_t *sizes, uint64_t count)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tag | 输入 | 缓存标识符，需与HcommAicpuTsTaskCacheLookup传入的tag一致。 |
| addrs | 输入 | 内存地址数组，前两项为用户输入/输出基址。V格式追加各rank的发送地址和接收地址，详见下文。 |
| sizes | 输入 | 与addrs一一对应的字节数。前两项为完整用户buffer大小，V格式后续项为各rank的字节count。 |
| count | 输入 | 普通格式为数组长度；V格式为`(UINT64_C(1) << 63) \| (2 + 2 * rankSize)`。 |

### V格式与字节单位

count的最高位为V格式标记，低63位表示实际数组长度，V格式的rankSize必须大于0。按rank顺序排列如下：

| 数组下标 | addrs | sizes |
| --- | --- | --- |
| 0 | input基址 | input完整字节数 |
| 1 | output基址 | output完整字节数 |
| 2 + r | input基址 + sdisplsBytes[r] | sendCountsBytes[r] |
| 2 + rankSize + r | output基址 + rdisplsBytes[r] | recvCountsBytes[r] |

HCOMM不感知算子data type。调用方必须将发送侧count/displ乘以sendType字节数，将接收侧count/displ乘以recvType字节数，检查乘法和地址加法溢出后再传入。HCOMM通过sizes恢复sendCounts/recvCounts，通过地址减去对应基址恢复sdispls/rdispls，四数组均以字节为单位。追加项仅为元数据，不作为远端内存段参与地址匹配。

非零count对应的范围必须位于完整用户buffer内。零count的偏移可以位于buffer外，但地址运算不得溢出；该项不表示实际内存访问。本接口录制缓存时，前两项基址必须非空；零数据侧在[Execute回放](HcommAicpuTsTaskCacheExecute.md#约束说明)时可使用空基址。

未设置最高位时，前两项为本地input/output，其余项按远端内存段处理，不根据数组长度推断V格式。

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

1. 仅支持AI CPU模式下，在Device侧调用该接口。
2. 仅在cache miss场景下调用（即HcommAicpuTsTaskCacheLookup返回isHit为false时），否则返回错误。
3. tag需与HcommAicpuTsTaskCacheLookup传入的tag一致，否则返回错误。
4. 该接口需与HcommAicpuTsTaskCacheEnd配套使用，在两者之间执行算子展开操作。
5. 如果AICPU Task Cache容量已满，不会添加新的cache entry，此时算子展开仍可正常执行但不会被缓存。
6. V格式需由调用方使用所有peer的发送和接收count均非零的预热算子录制完整任务模板，并通过准入保证每个远端peer最多一条有效channel、数据无需循环拆分、缓存复用期间通信资源保持不变。

## 调用示例

```c
const char *tag = "op_tag_example";
void *addrs[] = {inputAddr, outputAddr};
uint64_t sizes[] = {inputSize, outputSize};
uint64_t count = 2;

// cache miss下，开始缓存
if (HcommAicpuTsTaskCacheStart(tag, addrs, sizes, count) != HCCL_SUCCESS) {
    return 1;
}

// 执行算子展开（下发SQE/WQE等）
// ...

// 结束缓存
HcommAicpuTsTaskCacheEnd(tag);
```
