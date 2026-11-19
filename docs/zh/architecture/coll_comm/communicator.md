# Communicator

通信器模块，管理和协调集合通信操作。HCOMM 中每个 `HcclComm` 对应一个通信域上下文，内部由 `CollComm` 持有 RankGraph、本 rank 信息、通信引擎、通道、内存等资源。

> [!NOTE]说明
> 通信域静态缩容和扩容接口为试验特性，后续版本可能在不另行通知的情况下调整、重命名或移除，不承诺向后兼容。

## 职责

- 通信上下文管理
- 通信域创建、销毁与配置管理
- 集合操作创建与调度
- 拓扑信息维护
- 通信域静态扩缩容

## 通信域生命周期

通信域通常按照以下生命周期使用：

1. 通过rootInfo或rankTable方式创建初始通信域。
2. 在初始通信域上创建通信资源并执行集合通信。
3. 业务需要变更成员时，基于当前通信域执行静态缩容或扩容，得到新的通信域句柄。
4. 停止使用旧通信域，并在确认无在途任务后销毁旧通信域。
5. 业务结束时销毁当前通信域。

## 核心接口

### 扩缩容接口

本节仅列出通信域扩缩容相关接口。通信域创建、销毁、rank查询等基础接口参见[通信域创建与管理接口（C语言）](../../api_ref/comm_mgr_c/README.md)。

| 分类 | 接口 | 说明 |
| --- | --- | --- |
| 缩容 | [HcclCommShrink](../../api_ref/comm_mgr_c/HcclCommShrink.md) | 剔除指定rank，创建缩容后的新通信域。 |
| 扩容 | [HcclCommGrow](../../api_ref/comm_mgr_c/HcclCommGrow.md)、[HcclCommGetRootInfo](../../api_ref/comm_mgr_c/HcclCommGetRootInfo.md) | 获取扩容rootInfo，并加入新rank创建扩容后的新通信域。 |

## 静态扩缩容

### 缩容

缩容由所有存活rank调用[HcclCommShrink](../../api_ref/comm_mgr_c/HcclCommShrink.md)，从当前通信域中剔除指定rank。新通信域的rank编号按照原rank ID升序连续重排。

缩容不引入新rank，通常基于已有拓扑和通信域信息创建更小的通信域。如果被剔除的rank中包含root，新通信域在缩容完成后没有有效root；后续扩容前需要通过[HcclCommGetRootInfo](../../api_ref/comm_mgr_c/HcclCommGetRootInfo.md)触发root重选。

### 扩容

扩容由所有存活rank和新rank共同调用[HcclCommGrow](../../api_ref/comm_mgr_c/HcclCommGrow.md)。存活rank保持原rank编号，新rank从当前rank数量开始连续追加编号。

扩容前，由一个存活rank调用[HcclCommGetRootInfo](../../api_ref/comm_mgr_c/HcclCommGetRootInfo.md)获取`rootInfo`，并通过业务控制面分发给所有参与者。扩容需要进行拓扑信息收集和新建通信链路，因此所有参与者必须在同一轮扩容中完成调用。

### 派生通信域

扩缩容产生的通信域是当前通信域的派生通信域：

- 派生通信域持有新的通信域句柄和独立的通信资源。
- 父通信域的常规资源不会被原地改写。
- 通信域派生链只允许链尾通信域继续扩缩容。
- 持久化的拓扑探测资源由链尾派生通信域持有，用于后续继续扩缩容。

### 资源归属

扩缩容创建的新通信域拥有独立资源，旧通信域资源不会自动迁移：

- 对称内存窗口需要在新通信域上重新注册，并按需重新获取远端地址。
- CCU实例、CCU通道等资源需要基于新通信域重新查询或绑定。
- CCL Buffer、Channel、Thread、Engine Context等资源应在新通信域上重新获取或创建。
- 旧通信域需要由业务侧显式调用[HcclCommDestroy](../../api_ref/comm_mgr_c/HcclCommDestroy.md)销毁。

## 数据流

## 代码路径

- 通信域实现：`src/coll_communicator_mgr/communicator/coll_comm.cc`
- C接口适配：`src/coll_communicator_mgr/api_c_adpt/coll_comm_c_adpt.cc`
- 拓扑探测与派生：`src/coll_communicator_mgr/rank_info_detect/`
- 本端资源与通信引擎：`src/coll_communicator_mgr/resource_mgr/`
