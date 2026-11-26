# HcommResult

## 功能说明

HCOMM基础通信接口的返回值类型。

## 定义原型

```c
typedef int32_t HcommResult;

typedef enum {
    HCOMM_SUCCESS = 0,              /**< success */
    HCOMM_E_PARA = 1,               /**< parameter error */
    HCOMM_E_PTR = 2,                /**< empty pointer */
    HCOMM_E_MEMORY = 3,             /**< memory error */
    HCOMM_E_INTERNAL = 4,           /**< internal error */
    HCOMM_E_NOT_SUPPORT = 5,        /**< not support feature */
    HCOMM_E_NOT_FOUND = 6           /**< not found specific resource */
} HcommResultDef;
```

## 说明

HcommResult为int32_t类型，接口成功返回0，其他值表示失败。
