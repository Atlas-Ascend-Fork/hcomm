# CommMem

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-09-28T08:52:57.172Z pushedAt=2026-10-08T03:57:02.008Z -->

## Description

Structure describing memory segment metadata.

## Prototype

```c
typedef struct {
    CommMemType type; /* Memory physical location type. */
    void *addr;       /* Memory address. */
    uint64_t size;    /* Number of bytes in the memory region. */
} CommMem;
```
