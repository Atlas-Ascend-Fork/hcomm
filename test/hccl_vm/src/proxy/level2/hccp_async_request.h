/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef HCCP_ASYNC_REQUEST_H
#define HCCP_ASYNC_REQUEST_H

#include <cstdint>

// ===== hccp 异步请求最小公共状态（Defect 2 修复的公共结构）=====
// 合同（hccp_async.h）：*Async 提交成功返回 0 并给出唯一 req_handle，任务终态经
// RaGetAsyncReqResult 查询，查询即填写 AsyncReqResult 并释放句柄。
// 模拟器的异步操作在提交时同步完成，request 创建即终态。
// 所有生产 req_handle 的存根（RaSocket 族 / QP-jetty 族 / TP 族）必须统一经本
// 注册表登记唯一句柄，禁止共用假句柄 0x12345678（=十进制 305419896）——hcomm 的
// HrtRaGetAsyncReqResult 对未知句柄/查询失败会抛 NetworkApiException 中断通信域
// 初始化（CCU jetty 建链即受害者）。
namespace hccp_async_req {

// 模拟器本地接口标识（interfaceOpcode 透传保存；hcomm 侧仅对
// HCCP_OP_CTX_QP_CREATE(60) 有 jetty 资源耗尽特判，取远离该段的值避免歧义）。
constexpr unsigned int OP_SOCKET_RECV = 9001;
constexpr unsigned int OP_SOCKET_SEND = 9002;
constexpr unsigned int OP_SOCKET_BATCH_CONNECT = 9003;
constexpr unsigned int OP_SOCKET_LISTEN_STOP = 9004;
constexpr unsigned int OP_SOCKET_BATCH_CLOSE = 9005;
constexpr unsigned int OP_QP_IMPORT = 9101;
constexpr unsigned int OP_QP_UNIMPORT = 9102;
constexpr unsigned int OP_LMEM_UNREGISTER = 9103;
constexpr unsigned int OP_QP_DESTROY = 9104;
constexpr unsigned int OP_TP_INFO_LIST = 9105;
constexpr unsigned int OP_TP_ATTR_GET = 9106;
constexpr unsigned int OP_QP_CREATE = 9107;
constexpr unsigned int OP_TP_ATTR_SET = 9108;

// 登记一次性 request（reqResult 为任务终态：0 成功/SOCK_EAGAIN 重投/其他失败），
// 返回唯一句柄；分配失败返回 nullptr（调用方须以提交失败返回，不得返回空句柄）。
void* Register(int reqResult, unsigned int interfaceOpcode);

// 查询并释放：成功时填写 *reqResult/*interfaceOpcode 并释放句柄，返回 0；
// 句柄为空、未登记或已释放（重复查询）返回 -1，出参不动。
int Consume(void* reqHandle, int* reqResult, unsigned int* interfaceOpcode);

} // namespace hccp_async_req

#endif
