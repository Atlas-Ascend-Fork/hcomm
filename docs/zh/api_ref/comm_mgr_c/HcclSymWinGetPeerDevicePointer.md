# HcclSymWinGetPeerDevicePointer

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
<!-- npu="310p" id4 -->
- Atlas推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas训练系列产品：不支持
<!-- end id5 -->

## 功能说明

根据对称内存窗口资源句柄、偏移和对端标识，在HOST侧获取对称内存窗口中对应的对端Device地址指针。

<!-- npu="950" id8 -->
针对Ascend 950PR&950DT系列产品的UB Memory场景，对端标识为LSA WorldTeam的member ID，接口返回该LSA成员在对称VA空间中的Device地址。本接口在HOST侧调用。URMA场景不支持本接口，应使用[HcclSymWinGetRemoteAddr](HcclSymWinGetRemoteAddr.md)获取远端地址。
<!-- end id8 -->

<!-- npu="A3" id6 -->
针对Atlas A3系列产品，对称内存窗口未绑定UB Memory布局，本接口返回HCCL_E_NOT_SUPPORT并将*ptr置为nullptr，此时应改用[HcclSymWinGetPeerPointer](HcclSymWinGetPeerPointer.md)获取地址。
<!-- end id6 -->

## 函数原型

```c
HcclResult HcclSymWinGetPeerDevicePointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerMemberId, void** ptr)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| winHandle | 输入 | 对称内存窗口资源句柄。<br>HcclCommSymWindow类型的定义可参见[HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md)。 |
| offset | 输入 | 使用[HcclCommSymWinGet](HcclCommSymWinGet.md)获取到的偏移量，取值范围为[0, 窗口用户size)。 |
| peerMemberId | 输入 | 对端标识，为LSA WorldTeam的member ID，取值范围为[0, lsaTeamSize)。<br>该参数不是通信域rank ID，可通过[HcclTeamRankToMember](../comm_opdev/control_plane_api/comms_domain_resource_mgmt/HcclTeamRankToMember.md)转换得到。 |
| ptr | 输出 | 指向“对称内存窗口中对应地址”的指针。接口失败时将*ptr置为nullptr。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：

| 返回值 | 描述 |
| --- | --- |
| HCCL_SUCCESS | 获取对端地址成功。 |
| HCCL_E_PTR | winHandle或ptr为空指针。 |
| HCCL_E_NOT_SUPPORT | 窗口不是UB Memory对称内存窗口（如URMA场景），调用方需自行回退。 |
| HCCL_E_NOT_FOUND | winHandle未注册或已解注册。 |
| HCCL_E_PARA | peerMemberId越界，或offset超出窗口范围。 |

## 约束说明

<!-- npu="950" id9 -->
- 针对Ascend 950PR&950DT系列产品，UB Memory场景支持，URMA场景不支持，输入的peerMemberId必须是LSA WorldTeam的member ID。
- UB Memory场景会在运行时校验member ID，超出`[0, lsaTeamSize)`时返回`HCCL_E_PARA`。
- UB Memory场景会在运行时校验offset，超出`[0, 窗口用户size)`时返回`HCCL_E_PARA`。
<!-- end id9 -->
- 该接口在HOST侧调用，返回对端Device地址。
- 本接口符号由`libhcomm.so`提供。设备侧库`libccl_kernel.so`与`libaicpu_custom.so`不导出该符号。调用方应链接`libhcomm.so`，或通过`dlsym`从`libhcomm.so`加载本接口符号，不得从设备侧库中查找。
- 接口失败时*ptr一定为nullptr，调用方不得沿用上一次查询得到的地址。
- winHandle对应的窗口解注册（[HcclCommSymWinDeregister](HcclCommSymWinDeregister.md)）后，通过本接口获取的地址失效。

## 调用示例

HOST侧注册完对称内存window以后，在HOST侧按LSA WorldTeam member ID逐个获取对端Device地址，以下是伪代码描述：

```c
// comm、addr、size、peerRank均由调用方准备：comm为已创建的通信域，addr和size为本rank已注册为对称内存的地址和大小，peerRank为对端rank ID
// 获取本rank已注册的对称内存窗口及其偏移
HcclCommSymWindow temp_win = nullptr;
size_t offset = 0;
HcclCommSymWinGet(comm, addr, size, &temp_win, &offset);

// 通信域rank ID转换为LSA WorldTeam member ID
HcommTeamHandle lsaTeam = nullptr;
HcclTeamGetLsaTeam(comm, &lsaTeam);
uint32_t peerMemberId = 0;
HcclTeamRankToMember(comm, lsaTeam, peerRank, &peerMemberId);

// 获取对端地址，失败时peerPtr为nullptr
void *peerPtr = nullptr;
HcclResult ret = HcclSymWinGetPeerDevicePointer(temp_win, offset, peerMemberId, &peerPtr);
if (ret == HCCL_E_NOT_SUPPORT) {
    // 非UB Memory对称窗口，回退其他取址方式
}

// HOST侧获取到的peerPtr为对端Device地址
```
