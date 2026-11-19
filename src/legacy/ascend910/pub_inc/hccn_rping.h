/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCN_RPING_H_
#define HCCN_RPING_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void* HccnRpingCtx;

#define HCCN_RPING_PAYLOAD_LEN_MAX 1500

typedef enum {
    HCCN_SUCCESS = 0, /* success */
    HCCN_E_AGAIN,     /* try again */
    HCCN_E_FAIL,      /* fail */
    HCCN_E_PARA,      /* wrong parameter */
    HCCN_E_MEM,       /* memory */
    HCCN_E_RESERVED   /* reserved */
} HccnResult;

typedef enum {
    HCCN_RPING_MODE_ROCE = 0, /* RoCE */
    HCCN_RPING_MODE_UB = 1,   /* UB, default mode is UM + UTP */
    PING_MODE_IPoUB = 2,      /* UBoE */
    HCCN_RPING_MODE_RESERVED  /* reserved */
} HccnRpingMode;

#define HCCN_RPING_ADDR_TYPE uint32_t
#define HCCN_RPING_ADDR_TYPE_IP ((HCCN_RPING_ADDR_TYPE)0)
#define HCCN_RPING_ADDR_TYPE_EID ((HCCN_RPING_ADDR_TYPE)1)

typedef enum {
    HCCN_RPING_ADDTARGET_STATE_DONE = 0, /* add success */
    HCCN_RPING_ADDTARGET_STATE_DOING,    /* adding */
    HCCN_RPING_ADDTARGET_STATE_FAIL,     /* add fail */
    HCCN_RPING_ADDTARGET_STATE_TIMEOUT,  /* connect target timeout */
    HCCN_RPING_ADDTARGET_STATE_RESERVED  /* reserved */
} HccnRpingAddTargetState;

typedef enum {
    HCCN_RPING_RESULT_STATE_NOT_FOUND = 0, /* not found */
    HCCN_RPING_RESULT_STATE_INVALID,       /* invalid */
    HCCN_RPING_RESULT_STATE_VALID,         /* valid */
    HCCN_RPING_RESULT_STATE_RESERVED       /* reserved */
} HccnRpingResultState;

typedef struct HccnRpingInitAttrDef {
    HccnRpingMode mode;  /* Link type: RoCE : HCCN_RPING_MODE_ROCE /UB/ others */
    uint32_t port;       /* Port to listen when device being target */
    uint32_t npuNum;     /* Numbers of all the devices in the net */
    uint32_t bufferSize; /* Size of resource that device need to allocate when device being client */
    uint32_t sl;         /* service level, range: 0~7, need set as 4 when no use */
    uint32_t tc;         /* traffic class, range: 0~255, need set as 132 when no use */
    union {
        char* ipAddr; /* IP address of device */
        char* eid;    /* Eid of device */
    };
} HccnRpingInitAttr;

typedef struct HccnRpingTargetInfoDef {
    uint32_t srcPort; /* udp src port, hash lag needed */
    union {
        uint32_t reserved;
        uint32_t addrType; /* address type, HCCN_RPING_ADDR_TYPE_IP: ip, HCCN_RPING_ADDR_TYPE_EID: eid */
    };
    uint32_t sl;   /* service level, range: 0~7, need set as 4 when no use */
    uint32_t tc;   /* traffic class, range: 0~255, need set as 132 when no use */
    uint32_t port; /* port to connect target */
    uint32_t payloadLen;
    char payload[HCCN_RPING_PAYLOAD_LEN_MAX]; /* user defined payload */
    union {
        char* srcIp;  /* local(client) ip */
        char* srcEid; /* local(client) eid */
    };
    union {
        char* dstIp;  /* remote(target) ip */
        char* dstEid; /* remote(target) eid */
    };
} HccnRpingTargetInfo;

typedef struct HccnRpingResultInfoDef {
    uint32_t txPkt;             /* send pkt num */
    uint32_t rxPkt;             /* receive pkt num */
    uint32_t minRTT;            /* minimum round-trip time / usec */
    uint32_t maxRTT;            /* maximum round-trip time / usec */
    uint32_t avgRTT;            /* average round-trip time / usec */
    HccnRpingResultState state; /* ping result state: valid | invalid */
    uint32_t reserved[5U];
} HccnRpingResultInfo;

typedef struct HccnRpingTimestampDef {
    uint64_t sec;  /* time sec */
    uint64_t usec; /* time usec */
} HccnRpingTimestamp;

typedef struct HccnRpingAddTargetConfigDef {
    uint32_t connectTimeout; // 1ms-3600000ms
    uint8_t reserve[32];
} HccnRpingAddTargetConfig;

/**
 * @brief struct of every payload header
 */
typedef struct HccnRpingPayloadHeadDef {
    union {
        char srcIp[64];  /* local(client) ip */
        char srcEid[16]; /* local(client) eid */
    };
    union {
        char dstIp[64];  /* remote(target) ip  */
        char dstEid[16]; /* remote(target) eid */
    };
    uint32_t payloadLen; /* user defined payload length */
    uint32_t resvd[3];
    HccnRpingTimestamp t1; /* client send timestamp */
    HccnRpingTimestamp t2; /* target recv timestamp */
    HccnRpingTimestamp t3; /* target send timestamp */
    HccnRpingTimestamp t4; /* client recv timestamp */
    uint32_t rpingBatchId; /* batch ping task id */
    uint32_t addr_type;    /* address type, 0: ip, 1: eid */
    uint8_t reserved[40];  // PPT40
} HccnRpingPayloadHead;

// UB transaction mode: 事务模式
typedef enum {
    HCCN_RPING_UB_RM = 0x1,      /* reserved, not supported by HccnRpingInitWithCfg */
    HCCN_RPING_UB_RC = 0x1 << 1, /* Reliable connection */
    HCCN_RPING_UB_UM = 0x1 << 2  /* Unreliable message */
} HccnUbTransactionMode;

// UB transport mode: 传输平面模式
typedef enum {
    HCCN_RPING_UB_RTP = 0, /* Reliable transport plane */
    HCCN_RPING_UB_CTP,     /* Connectionless transport plane */
    HCCN_RPING_UB_UTP      /* Unreliable transport plane */
} HccnUbTransportMode;

// 初始化配置结构体:分离事务模式与传输平面模式
typedef struct HccnRpingInitCfgDef {
    uint32_t validCfgLen; /* length of valid cfg, must be set to sizeof(HccnRpingInitCfg) by caller */
    union {
        struct {
            HccnUbTransactionMode taMode; /* UB transaction mode */
            HccnUbTransportMode tpMode;   /* UB transport mode */
            uint32_t reserved[64];        /* reserved */
        } ub;
    };
} HccnRpingInitCfg;

// DFX 诊断接口:获取 rping 上下文信息(如本端 jettyId)
typedef enum {
    HCCN_RPING_CTX_INFO_JETTY = 0, /* jetty ctx info */
    HCCN_RPING_CTX_INFO_RESERVED
} HccnRpingCtxInfoType;

typedef void* HccnRpingCtxInfo;

typedef struct HccnRpingCtxJettyInfoDef {
    uint32_t jettyId;      /* local jettyId */
    uint32_t reserved[63]; /* reserved for extension */
} HccnRpingCtxJettyInfo;

/**
 * @brief Get rping context info (e.g. local jettyId) after HccnRpingInit / HccnRpingInitWithCfg.
 *        Only UB mode (HCCN_RPING_MODE_UB) is supported;
 *        RoCE mode or an uninitialized context will return HCCN_E_FAIL.
 * @param rpingCtx : context of rping resource (HccnRpingCtx is void*, so rpingCtx is void**).
 * @param infoType : info type, only HCCN_RPING_CTX_INFO_JETTY supported now.
 * @param info : output buffer pointer's address (HccnRpingCtxInfo is void*, so info is void**),
 *               caller should pre-allocate HccnRpingCtxJettyInfo sized memory.
 * @param infoLen : info buffer length, must be >= sizeof(HccnRpingCtxJettyInfo).
 * @return HccnResult
 */
extern HccnResult
HccnRpingGetCtxInfo(HccnRpingCtx* rpingCtx, HccnRpingCtxInfoType infoType, HccnRpingCtxInfo* info, uint32_t infoLen);

/**
 * @brief Init rping resource on a device.
 * @param devLogicId : Device logic ID.
 * @param initAttr: init attribute.
 * @param rpingCtx: context of rping resource.
 * @return HccnResult
 */
extern HccnResult HccnRpingInit(uint32_t devLogicId, HccnRpingInitAttr* initAttr, HccnRpingCtx* rpingCtx);

/**
 * @brief Init rping resource on a device with cfg (support RC+CTP).
 *        老接口 HccnRpingInit 保持不变, 新接口通过initCfg分离传入事务模式与传输平面模式。
 * @param devLogicId : Device logic ID.
 * @param initAttr: init attribute, mode 应设为 HCCN_RPING_MODE_UB。
 * @param initCfg: init config, ub.taMode/tpMode 决定具体组合(如 RC+CTP)。
 *                 validCfgLen 必须设置为 sizeof(HccnRpingInitCfg)，用于前向兼容校验，
 *                 小于该值时 HccnRpingInitWithCfg 返回 HCCN_E_PARA。
 * @param rpingCtx: context of rping resource.
 * @return HccnResult
 */
extern HccnResult HccnRpingInitWithCfg(
    uint32_t devLogicId, HccnRpingInitAttr* initAttr, HccnRpingInitCfg* initCfg, HccnRpingCtx* rpingCtx);
/**
 * @brief Release rping resource on a device.
 * @param rpingCtx: context of rping resource.
 * @return HccnResult
 */
extern HccnResult HccnRpingDeinit(HccnRpingCtx rpingCtx);

/**
 * @brief Add targets to client.
 * @param rpingCtx: context of rping resource.
 * @param targetNum: Number of NPUs need probe.
 * @param target: Infoes of NPU need probe, this is an array.
 * @return HccnResult
 * @note RC mode (ctx created by HccnRpingInitWithCfg with ub.taMode=HCCN_RPING_UB_RC) only supports
 *       1 target per add call.
 */
extern HccnResult HccnRpingAddTarget(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target);

/**
 * @brief Add targets to client.
 * @param rpingCtx: context of rping resource.
 * @param targetNum: Number of NPUs need probe.
 * @param target: Infoes of NPU need probe, this is an array.
 * @param config: HccnRpingAddTargetConfig.
 * @return HccnResult
 */
extern HccnResult HccnRpingAddTargetV2(
    HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target,
    HccnRpingAddTargetConfig* config); // deprecated

/**
 * @brief Add targets to client.
 * @param rpingCtx: context of rping resource.
 * @param targetNum: Number of NPUs need probe.
 * @param target: Infoes of NPU need probe, this is an array.
 * @param config: HccnRpingAddTargetConfig.
 * @return HccnResult
 * @note RC mode (ctx created by HccnRpingInitWithCfg with ub.taMode=HCCN_RPING_UB_RC) only supports
 *       1 target per add call.
 */
extern HccnResult HccnRpingAddTargetWithCfg(
    HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target, HccnRpingAddTargetConfig* config);

/**
 * @brief Remove targets from targets.
 * @param rpingCtx: context of rping resource.
 * @param targetNum: Number of NPUs need probe.
 * @param target: Infoes of NPU need probe, this is an array.
 * @return HccnResult
 */
extern HccnResult HccnRpingRemoveTarget(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target);

/**
 * @brief Get adding target's state.
 * @param rpingCtx: context of rping resource.
 * @param targetNum: Number of NPUs need probe.
 * @param target: Infoes of NPU need probe, this is an array.
 * @param targetState: target state, this is an array.
 * @return HccnResult
 */
extern HccnResult HccnRpingGetTarget(
    HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target, HccnRpingAddTargetState* targetState);

/**
 * @brief Start batch ping task.
 * @param rpingCtx: context of rping resource.
 * @param pktNum: Number of packet send to target.
 * @param interval: Interval between two sends of ping packet.
 * @param timeout: Time threshold between ping & pong packet.
 * @return HccnResult
 */
extern HccnResult HccnRpingBatchPingStart(HccnRpingCtx rpingCtx, uint32_t pktNum, uint32_t interval, uint32_t timeout);

/**
 * @brief Stop batch ping task.
 * @param rpingCtx: context of rping resource.
 * @return HccnResult
 */
extern HccnResult HccnRpingBatchPingStop(HccnRpingCtx rpingCtx);

/**
 * @brief Get batch ping results.
 * @param rpingCtx: context of rping resource.
 * @param targetNum: Number of NPUs need probe.
 * @param target: Infoes of NPU need probe, this is an array.
 * @param result: probe result, this is an array.
 * @return HccnResult
 */
extern HccnResult
HccnRpingGetResult(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target, HccnRpingResultInfo* result);

/**
 * @brief Get batch ping packet payload.
 * @param rpingCtx: context of rping resource.
 * @param payload: packet payload pointer, contain all payload, every payload head struct as HccnRpingPayloadHeader.
 * @param payloadLen: length of all payload.
 * @return HccnResult
 */
extern HccnResult HccnRpingGetPayload(HccnRpingCtx rpingCtx, void** payload, uint32_t* payloadLen);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCN_RPING_H_
