# HcommMemHostUnmapFromDev

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas训练系列产品：不支持
<!-- end id5 -->

## 功能说明

解除`HcommMemHostMap2Dev`建立的Host DDR内存到Device侧虚拟地址（dVA）的映射，释放对应的dVA。

## 函数原型

```c
HcommResult HcommMemHostUnmapFromDev(const CommMem* hostMem)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| hostMem | 输入 | Host侧内存描述，需与`HcommMemHostMap2Dev`的入参一致。`type`必须为`COMM_MEM_TYPE_HOST`，`addr`为Host侧虚拟地址（hVA）。 |

## 返回值

[HcommResult](../comm_opdev/datatype_definition/HcommResult.md)：返回 `HCCL_SUCCESS` 表示接口调用成功；其他值表示失败。

- `HCCL_E_PTR`：`hostMem` 为空指针。
- `HCCL_E_PARA`：`hostMem` 参数非法（`type` 不是`COMM_MEM_TYPE_HOST`或`addr` 为空指针）。
- `HCCL_E_RUNTIME`：获取当前Device、获取芯片类型、获取Host到Device（H2D）协议或Host内存注销接口调用失败。
- `HCCL_E_NOT_SUPPORT`：当前Runtime版本不支持`rtHostUnRegisterByPciAtu`，H2D为PCIe协议且产品为Ascend 950PR&950DT系列产品时无法完成Host内存注销。

## 约束说明

- 调用前需通过`aclrtSetDevice`设置当前Device，接口内部通过`aclrtGetDevice`获取当前设备逻辑ID。
- `HcommMemHostMap2Dev`和`HcommMemHostUnmapFromDev`必须在同一Device上下文（同一当前Device）中调用。
- 必须与`HcommMemHostMap2Dev`成对调用，`hostMem`需与映射时的入参一致。
- 注销成功后，映射得到的dVA失效，不能再被Device侧访问。
- 重复注销或注销未映射的地址时，由Runtime返回错误。
- H2D为PCIe协议且产品为Ascend 950PR&950DT系列产品时使用`rtHostUnRegisterByPciAtu`，其他产品使用ACL的Host内存注销接口（`aclrtHostUnregister`）；两种方式均以Host侧地址（hVA）为入参。
- 当Runtime不支持H2D连接类型查询（例如老版本Runtime）时，接口按非PCIe处理，回退`aclrtHostUnregister`注销路径。

## 调用示例

```c
// 返回值检查宏
#define HCOMMCHECK(cmd) do { HcommResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

size_t memSize = 4096;
void *rawHostPtr = nullptr;
ACLCHECK(aclrtMallocHost(&rawHostPtr, memSize + 4096));
uintptr_t aligned = (reinterpret_cast<uintptr_t>(rawHostPtr) + 4095) & ~uintptr_t(4095);
void *hostPtr = reinterpret_cast<void *>(aligned);

CommMem hostMem;
hostMem.type = COMM_MEM_TYPE_HOST;
hostMem.addr = hostPtr;
hostMem.size = memSize;

CommMem devMem;
devMem.type = COMM_MEM_TYPE_INVALID;
devMem.addr = nullptr;
devMem.size = 0;
HCOMMCHECK(HcommMemHostMap2Dev(&hostMem, &devMem, HCOMM_MEM_MAP_DEFAULT));

// 使用完成后解除映射
HCOMMCHECK(HcommMemHostUnmapFromDev(&hostMem));

ACLCHECK(aclrtFreeHost(rawHostPtr));
```
