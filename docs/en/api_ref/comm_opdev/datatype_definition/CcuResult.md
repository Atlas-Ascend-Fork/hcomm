# CcuResult

<!-- md-trans-meta sourceCommit=7d6d6e1bcb7f9eb88aaeb10e2c4e1e8360a910b0 translatedAt=2026-09-28T08:51:00.143Z pushedAt=2026-10-08T03:53:04.312Z -->

## Description

Return value type of CCU APIs, used to indicate the execution result of an operation. All CCU control-plane and data-plane APIs return the operation status using this type.

## Prototype

```c
typedef enum {
    CCU_SUCCESS = 0,
    CCU_E_PARA = 1,
    CCU_E_PTR = 2,
    CCU_E_INTERNAL = 4,
    CCU_E_NOT_SUPPORT = 5,
    CCU_E_NOT_FOUND = 6,
    CCU_E_UNAVAIL = 7,
    CCU_E_RUNTIME = 15,
    CCU_E_DRV_START = 4096,
    CCU_E_DRV_INIT_FAILED = 4097,
    CCU_E_DRV_BUSY = 4098,
    CCU_E_DRV_END = 4224,
    CCU_E_RESERVED = 9216
} CcuResult;
```

## Enum Value Description

| Enum Value | Value | Description |
| --- | --- | --- |
| `CCU_SUCCESS` | 0 | Operation succeeded. |
| `CCU_E_PARA` | 1 | Invalid parameter. |
| `CCU_E_PTR` | 2 | Null pointer error. |
| `CCU_E_INTERNAL` | 4 | Internal error. |
| `CCU_E_NOT_SUPPORT` | 5 | Unsupported feature. |
| `CCU_E_NOT_FOUND` | 6 | Specified resource not found. |
| `CCU_E_UNAVAIL` | 7 | Resource unavailable. |
| `CCU_E_RUNTIME` | 15 | Runtime error. |
| `CCU_E_DRV_START` | 4096 | Start marker of the driver-layer error code range (not returned as an actual error code). |
| `CCU_E_DRV_INIT_FAILED` | 4097 | Driver initialization failed. |
| `CCU_E_DRV_BUSY` | 4098 | Driver busy. |
| `CCU_E_DRV_END` | 4224 | End marker of the driver-layer error code range (not returned as an actual error code). |
| `CCU_E_RESERVED` | 9216 | Start marker of the reserved error code range (not returned as an actual error code; reserved for future extension). |
