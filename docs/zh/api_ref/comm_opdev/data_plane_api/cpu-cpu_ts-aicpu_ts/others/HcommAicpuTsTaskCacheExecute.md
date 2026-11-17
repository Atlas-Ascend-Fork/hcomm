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
| addrs | 输入 | 新的内存地址数组，布局与[Start](HcommAicpuTsTaskCacheStart.md)相同。非V格式前两项为本端input/output，携带远端内存时`addrs[2 + 2 * r]`、`addrs[3 + 2 * r]`分别为rank r的input/output基址。V格式前两项为input/output，其后为发送、接收各rank地址。两种格式均允许基址数值为0；数组指针本身不能为空。 |
| sizes | 输入 | 与addrs对应的字节数。非V格式各项须与录制时一致；V格式追加项用于恢复sendCounts/recvCounts，地址减去相应基址恢复字节单位的sdispls/rdispls。 |
| count | 输入 | 与Start相同的格式和长度。非V格式为数组长度：仅本端内存时为2，携带远端内存时为`2 + 2 * rankSize`；奇数且大于2时返回HCCL_E_PARA，count小于2或与录制时长度不一致时返回HCCL_E_INTERNAL。V格式为`(UINT64_C(1) << 63) \| (2 + 2 * rankSize)`。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

V格式支持在完整任务模板上刷新count、displ和用户buffer大小，不因这些参数变化返回HCCL_E_NOT_SUPPORT。不支持的设备仍返回HCCL_E_NOT_SUPPORT；非法地址、范围或V参数返回HCCL_E_PARA，缓存元数据损坏或count/非V sizes与录制不一致返回HCCL_E_INTERNAL。回放失败时不保证所有任务均未下发，调用方应传播错误，不应从头重新展开本次算子。

## 约束说明

1. 仅支持AI CPU模式下，在Device侧调用该接口。
2. 仅在cache hit场景下调用（即HcommAicpuTsTaskCacheLookup返回isHit为true时），否则返回错误。
3. tag需与HcommAicpuTsTaskCacheLookup传入的tag一致，否则返回错误。
4. V格式由调用方通过全peer非零预热录制完整任务模板，并保证每个远端peer最多一条有效channel、数据无需循环拆分、缓存复用期间通信资源保持不变。在此范围内支持刷新基址、count、displ和完整buffer大小；rankSize须与录制时一致。
5. 内部RefreshAndLaunch返回后会重置本次调用的cache上下文，保留缓存条目；设备、接口参数或V格式解析等入口检查提前失败时，不执行该重置。
6. HCOMM不感知data type，调用方须在传入前完成元素数到字节数的转换，并遵守Start的范围和溢出约束。非V格式要求count及各项sizes与录制时一致；携带远端内存时，槽位按rank编号成对排列，详见[Start非V格式与远端内存布局](HcommAicpuTsTaskCacheStart.md#非v格式与远端内存布局)。
7. V格式回放允许基址为0，包括该侧存在非零count的情况；非零count对应的范围仍须位于完整buffer内，地址加法不得溢出。零count项可保留非零displ；零数据任务回放为NOP或占位任务，带通知的WRITE保留同步通知并清空数据SGE。
8. 普通格式和V格式均以本次基址加偏移计算刷新地址，基址为0时结果即为偏移数值，不会自动补上其他地址。调用方须保证最终访问的地址及通信资源有效；普通格式仍要求回放长度与录制一致。
9. 非V格式支持input/output完全同址或部分重叠，调用方须保证算子和算法支持该布局。重叠布局以三项信息描述：重叠字节数、重叠起点相对input的偏移、重叠起点相对output的偏移。调用方生成tag时须区分不同重叠布局，使其不会复用同一缓存条目；HCOMM将tag作为不透明标识符，不解析其中的布局信息。回放前，本端input/output以及携带远端内存时各rank的input/output均会逐对校验这三项是否与录制时一致。不重叠或任一侧大小为0时，三项均按0处理，允许两侧独立换址。布局不一致时，在刷新和下发任务前返回HCCL_E_PARA，不会自动回退重新展开。本约束不扩展V格式的inplace支持范围。
10. 非V格式录制和回放须采用相同的本端内存或携带远端内存布局，二者不一致时返回HCCL_E_PARA。携带远端内存的缓存条目在录制阶段通过channel查询并保存对端rank（查询回调由HcommAcquireComm内部注册，调用方无注册接口；录制须在同一线程的HcommAcquireComm与HcommReleaseComm之间完成，且录制前须确保channel已建链并完成对端rank记录）；回放沿用录制时保存的rank归属，不再查询回调。

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
