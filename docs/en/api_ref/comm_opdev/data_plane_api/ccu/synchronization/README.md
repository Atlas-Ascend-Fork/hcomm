# Introduction

<!-- md-trans-meta sourceCommit=cc5c3b8879863bf96ae5d23850c6b95c814dd0ed translatedAt=2026-09-28T08:20:22.914Z pushedAt=2026-09-30T09:16:15.963Z -->

This section provides the synchronization APIs in the CCU kernel for coordinating the completion order of asynchronous operations through events and Notify resources.

In the CCU kernel, data movement and cross-kernel stage execution advance asynchronously. When an "order relationship" needs to be explicitly established between different kernels or different movement operations, the production side calls the **Record** API to write a completion signal, and the consumption side calls the **Wait** API to block and wait for the signal to arrive. Based on the positional relationship between the producer and the consumer, synchronization is classified into the following three types:

| Type | Applicable Scenario | Production-Side API | Consumption-Side API |
| --- | --- | --- | --- |
| Local Event | Same kernel: waits for the completion of an asynchronous movement initiated within this kernel | [EventRecord](EventRecord.md) | [EventWait](EventWait.md) |
| Local Notify | Cross-kernel within the same die: coordinates the order between different kernels, paired by a string tag | [LocalNotifyRecord](LocalNotifyRecord.md) | [LocalNotifyWait](LocalNotifyWait.md) |
| Remote Notify | Cross-die (including cross-device and cross-node): transmits signals through a channel | [NotifyRecord](NotifyRecord.md) | [NotifyWait](NotifyWait.md) |

[WriteVariableWithNotify](WriteVariableWithNotify.md) is an extended API of remote Notify. It combines "writing a remote variable value" and "triggering a remote Notify" into an atomic operation, and is used in scenarios where a scalar value needs to be sent to the peer together with a completion signal.

## API List

- [EventRecord](EventRecord.md)
- [EventWait](EventWait.md)
- [LocalNotifyRecord](LocalNotifyRecord.md)
- [LocalNotifyWait](LocalNotifyWait.md)
- [NotifyRecord](NotifyRecord.md)
- [NotifyWait](NotifyWait.md)
- [WriteVariableWithNotify](WriteVariableWithNotify.md)
