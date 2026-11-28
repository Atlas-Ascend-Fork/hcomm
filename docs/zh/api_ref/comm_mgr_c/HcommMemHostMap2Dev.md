# HcommMemHostMap2Dev

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

将Host侧DDR内存映射为Device侧可访问的虚拟地址（dVA），便于Device侧直接读写该Host内存。

## 函数原型

```c
HcommResult HcommMemHostMap2Dev(const CommMem* hostMem, CommMem* devMem, HcommMemMapType mapType)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| hostMem | 输入 | Host侧内存描述。`type`必须为`COMM_MEM_TYPE_HOST`，`addr`为Host侧虚拟地址（hVA），`size`为内存大小（字节），不能为0。 |
| devMem | 输出 | 映射结果。接口调用成功后，`addr`为Device侧可访问的dVA，`type`为`COMM_MEM_TYPE_DEVICE`，`size`与`hostMem->size`一致。 |
| mapType | 输入 | Host内存映射方式，参见`HcommMemMapType`：<br>`HCOMM_MEM_MAP_DEFAULT`：将Host内存映射注册为Device可访问的内存地址（默认）。<br>`HCOMM_MEM_MAP_PINNED`：将Host上通过C/C++标准库函数（例如`malloc`）申请的非锁页内存转为锁页内存后再映射；仅非ATU（ACL）路径支持，ATU场景返回`HCCL_E_NOT_SUPPORT`。 |

## 返回值

[HcommResult](../comm_opdev/datatype_definition/HcommResult.md)：返回 `HCCL_SUCCESS` 表示接口调用成功；其他值表示失败。

- `HCCL_E_PTR`：`hostMem` 或 `devMem` 为空指针。
- `HCCL_E_PARA`：`hostMem` 参数非法（`type` 不是 `COMM_MEM_TYPE_HOST`、`addr` 为空指针或 `size` 为 0），或 `mapType` 取值非法。
- `HCCL_E_RUNTIME`：获取当前Device、获取芯片类型、获取H2D协议、Host内存注册接口或`aclrtHostGetDevicePointer`调用失败。
- `HCCL_E_NOT_SUPPORT`：`mapType`为`HCOMM_MEM_MAP_PINNED`但走ATU路径，或当前Runtime版本不支持`rtHostRegisterByPciAtu`/`rtHostUnRegisterByPciAtu`，H2D为PCIe协议且产品为Ascend 950PR&950DT系列产品时无法完成Host内存注册。

## 约束说明

- 调用前需通过`aclrtSetDevice`设置当前Device，接口内部通过`aclrtGetDevice`获取当前设备逻辑ID。
- `HcommMemHostMap2Dev`和`HcommMemHostUnmapFromDev`必须在同一Device上下文（同一当前Device）中调用。
- `hostMem->addr`需4K页对齐。
- 当OS内核版本为5.10或更低时，必须使用`aclrtMallocHost`申请Host锁页内存，否则映射会异常；当OS内核版本高于5.10时，既支持使用`aclrtMallocHost`申请锁页内存，也支持使用`malloc`申请非锁页内存。
- `mapType`由HCOMM内部转换为底层注册标志：H2D为PCIe协议且产品为Ascend 950PR&950DT系列产品时使用`rtHostRegisterByPciAtu`，仅支持`HCOMM_MEM_MAP_DEFAULT`；其他产品使用`aclrtHostRegisterV2`，支持`HCOMM_MEM_MAP_DEFAULT`和`HCOMM_MEM_MAP_PINNED`。非ATU场景注册成功后通过`aclrtHostGetDevicePointer`获取dVA。
- 当Runtime不支持H2D连接类型查询（例如老版本Runtime）时，接口按非PCIe处理，回退`aclrtHostRegisterV2`注册路径。
- 映射得到的dVA供Device侧访问Host内存使用，不能用于内存拷贝操作。
- 映射得到的dVA仅在完成映射的进程及对应Device上下文内有效，不能跨进程或跨Device传递。
- Host内存的生命周期由调用方保证，映射期间不能释放或复用该内存；不再使用时需调用[HcommMemHostUnmapFromDev](./HcommMemHostUnmapFromDev.md)解除映射。
- 接口失败时不会修改`devMem`的内容。

## 调用示例

```c
// 返回值检查宏
#define HCOMMCHECK(cmd) do { HcommResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// 申请Host内存，并手动向上对齐到4K（aclrtMallocHost仅保证64B对齐）
size_t memSize = 4096;
void *rawHostPtr = nullptr;
ACLCHECK(aclrtMallocHost(&rawHostPtr, memSize + 4096));
uintptr_t aligned = (reinterpret_cast<uintptr_t>(rawHostPtr) + 4095) & ~uintptr_t(4095);
void *hostPtr = reinterpret_cast<void *>(aligned);

// 将Host内存映射为Device侧可访问的dVA
CommMem hostMem;
hostMem.type = COMM_MEM_TYPE_HOST;
hostMem.addr = hostPtr;
hostMem.size = memSize;

CommMem devMem;
devMem.type = COMM_MEM_TYPE_INVALID;
devMem.addr = nullptr;
devMem.size = 0;
HCOMMCHECK(HcommMemHostMap2Dev(&hostMem, &devMem, HCOMM_MEM_MAP_DEFAULT));

// 映射成功后，devMem.addr即为Device侧可访问的dVA

// 使用完成后解除映射
HCOMMCHECK(HcommMemHostUnmapFromDev(&hostMem));

ACLCHECK(aclrtFreeHost(rawHostPtr));
```
