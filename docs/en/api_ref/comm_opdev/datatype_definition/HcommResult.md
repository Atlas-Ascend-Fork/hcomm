# HcommResult

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T09:00:42.859Z pushedAt=2026-10-08T06:21:45.983Z -->

## Description

Return value type of the HCOMM basic communication APIs.

## Prototype

```c
typedef int32_t HcommResult;
```

## Description

**HcommResult** is of the **int32_t** type. The API returns **0** on success, and other values indicate failure. Its error codes are consistent with **HcclResult**. For the specific error code definitions, see [HcclResult](../../comm_mgr_c/data_type_definition/HcclResult.md).
