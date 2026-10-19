# Variable/Event资源预约

在CCU实例上预约物理编号连续的Variable（标量寄存器）或Event（完成事件单元）资源并获取预约句柄，再通过句柄查询段内各资源已缓存的虚拟地址，供Kernel内绑定或与CCU之外的目标模块共享。

- [HcommCcuVariableAlloc](HcommCcuVariableAlloc.md)
- [HcommCcuEventAlloc](HcommCcuEventAlloc.md)
- [HcommCcuVariableGetAddr](HcommCcuVariableGetAddr.md)
- [HcommCcuEventGetAddr](HcommCcuEventGetAddr.md)
