# HcclConfigValue

<!-- md-trans-meta sourceCommit=f93beaf76dbd6541b30a3081bdf2f9c63ab48e00 translatedAt=2026-09-28T06:36:48.219Z pushedAt=2026-09-28T08:22:34.096Z -->

## Description

Defines the value of a configurable parameter in [HcclConfig](HcclConfig.md).

## Prototype

```c
union HcclConfigValue {
    int32_t value;
};
```

**value** is the value of the **HCCL_DETERMINISTIC** parameter in [HcclConfig](HcclConfig.md).
