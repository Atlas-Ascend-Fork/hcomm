# HcommAicpuTsTaskCacheExecute

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

在cache hit场景下，根据新传入的内存地址信息刷新已缓存的task并下发执行。该接口会根据新的内存基址刷新SQE/WQE中的地址字段；V类复用录制时WQE中的token id/value，非V类按新地址刷新token信息，然后按下发顺序统一提交执行。

## 函数原型

```c
HcommResult HcommAicpuTsTaskCacheExecute(const char *tag, void **addrs, uint64_t *sizes, uint64_t count)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tag | 输入 | 缓存标识符，需与HcommAicpuTsTaskCacheLookup传入的tag一致。 |
| addrs | 输入 | 新的内存地址数组。V格式与[Start](HcommAicpuTsTaskCacheStart.md)相同，前两项为input/output，其后为发送、接收各rank地址。回放时，某侧所有peer的count均为0，允许该侧基址为空。 |
| sizes | 输入 | 与addrs对应的字节数。V格式追加项用于恢复sendCounts/recvCounts，地址减去相应基址恢复字节单位的sdispls/rdispls。 |
| count | 输入 | 与Start相同的格式和长度：普通格式为数组长度，V格式为`(UINT64_C(1) << 63) \| (2 + 2 * rankSize)`。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

V格式支持在完整任务模板上刷新count、displ和用户buffer大小，不因这些参数变化返回HCCL_E_NOT_SUPPORT。不支持的设备仍返回HCCL_E_NOT_SUPPORT；非法地址、范围或V参数返回HCCL_E_PARA，缓存元数据损坏返回HCCL_E_INTERNAL。回放失败时不保证所有任务均未下发，调用方应传播错误，不应从头重新展开本次算子。

## 约束说明

1. 仅支持AI CPU模式下，在Device侧调用该接口。
2. 仅在cache hit场景下调用（即HcommAicpuTsTaskCacheLookup返回isHit为true时），否则返回错误。
3. tag需与HcommAicpuTsTaskCacheLookup传入的tag一致，否则返回错误。
4. V格式由调用方通过全peer非零预热录制完整任务模板，并保证每个远端peer最多一条有效channel、数据无需循环拆分、缓存复用期间通信资源保持不变。在此范围内支持刷新基址、count、displ和完整buffer大小；rankSize须与录制时一致。
5. 内部RefreshAndLaunch返回后会重置本次调用的cache上下文，保留缓存条目；设备、接口参数或V格式解析等入口检查提前失败时，不执行该重置。
6. HCOMM不感知data type，调用方须在传入前完成元素数到字节数的转换，并遵守Start的范围和溢出约束。旧格式仍要求count和sizes与缓存一致。
7. V格式回放允许零数据侧使用空基址，包括零count项保留非零displ的情况。只要该侧任一peer的count非零，基址就必须非空。零数据任务回放为NOP或占位任务；带通知的WRITE保留同步通知并清空数据SGE。

## 调用示例

```c
const char *tag = "op_tag_example";
bool isHit = false;
void *addrs[] = {inputAddr, outputAddr};
uint64_t sizes[] = {inputSize, outputSize};
uint64_t count = 2;

// 查找cache
HcommAicpuTsTaskCacheLookup(tag, &isHit);

if (isHit) {
    // cache hit：刷新地址并下发
    if (HcommAicpuTsTaskCacheExecute(tag, addrs, sizes, count) != HCCL_SUCCESS) {
        return 1;
    }
} else {
    // cache miss：开始缓存
    HcommAicpuTsTaskCacheStart(tag, addrs, sizes, count);
    // 执行算子展开...
    HcommAicpuTsTaskCacheEnd(tag);
}
```
