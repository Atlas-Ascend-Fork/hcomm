# CommAbiHeader

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-09-28T08:51:35.283Z pushedAt=2026-10-08T03:55:07.452Z -->

## Description

Structure compatible with Abi fields.

## Prototype

```c
typedef struct {
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
} CommAbiHeader;
```
