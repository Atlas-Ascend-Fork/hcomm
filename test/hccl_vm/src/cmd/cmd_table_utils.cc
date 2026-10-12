/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <vector>

#include "cmd_base_utils.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "storage/table_access.h"
#include "runtime_state/db_sim_runner_ops.h"

namespace HcclSim {
template <typename T>
static void
PrintTable(const std::string& header, const std::vector<T>& rows, std::function<std::string(const T&)> formatter)
{
    std::cout << header << std::endl;
    for (const auto& row : rows) {
        std::cout << formatter(row) << std::endl;
    }
    std::cout << std::endl;
}

void CmdTableShow(std::string& tableName)
{
    if (tableName == "Device") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>());
        std::vector<sim::runtime::Device> tables = selected.value_or(std::vector<sim::runtime::Device>{});
        PrintTable<sim::runtime::Device>(
            "| id | server_id | user_id | logic_id | physical_id | super_device_id | overflow_mode | soc_version | "
            "status |",
            tables, [](const sim::runtime::Device& d) {
                return "| " + std::to_string(d.id) + " | " + std::to_string(d.server_id) + " | "
                       + std::to_string(d.user_id) + " | " + std::to_string(d.logic_id) + " | "
                       + std::to_string(d.physical_id) + " | " + std::to_string(d.super_device_id) + " | "
                       + std::to_string(d.overflow_mode) + " | " + std::string(d.soc_version) + " | "
                       + std::to_string(d.status) + " |";
            });
    } else if (tableName == "Server") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::Server>(HcclSim::Storage::All<sim::runtime::Server>());
        std::vector<sim::runtime::Server> tables = selected.value_or(std::vector<sim::runtime::Server>{});
        PrintTable<sim::runtime::Server>(
            "| id | pod_id | version | hardware_type |", tables, [](const sim::runtime::Server& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.pod_id) + " | "
                       + std::string(tmp.version) + " | " + std::string(tmp.hardware_type) + " |";
            });
    } else if (tableName == "Host") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::Host>(HcclSim::Storage::All<sim::runtime::Host>());
        std::vector<sim::runtime::Host> tables = selected.value_or(std::vector<sim::runtime::Host>{});
        PrintTable<sim::runtime::Host>("| id | server_id | ip | arch |", tables, [](const sim::runtime::Host& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.server_id) + " | "
                   + std::string(tmp.ip_addr) + " | " + std::to_string(tmp.arch) + " |";
        });
    } else if (tableName == "Runner") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::Runner>(HcclSim::Storage::All<sim::runtime::Runner>());
        std::vector<sim::runtime::Runner> tables = selected.value_or(std::vector<sim::runtime::Runner>{});
        PrintTable<sim::runtime::Runner>(
            "| id | host_id | pid | thread_id | timeout_config_ms | current_ctx_id |", tables,
            [](const sim::runtime::Runner& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.host_id) + " | "
                       + std::to_string(tmp.pid) + " | " + std::to_string(tmp.thread_id) + " | "
                       + std::to_string(tmp.timeout_config_ms) + " | " + std::to_string(tmp.current_ctx_id) + " |";
            });
    } else if (tableName == "Communicator") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(
            HcclSim::Storage::All<sim::runtime::Communicator>());
        std::vector<sim::runtime::Communicator> tables = selected.value_or(std::vector<sim::runtime::Communicator>{});
        PrintTable<sim::runtime::Communicator>(
            "| id | comm_id | comm_hash | rank_size | rank_id | device_id |", tables,
            [](const sim::runtime::Communicator& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::string(tmp.comm_id) + " | "
                       + std::to_string(tmp.comm_hash) + " | " + std::to_string(tmp.rank_size) + " | "
                       + std::to_string(tmp.rank_id) + " | " + std::to_string(tmp.device_id) + " |";
            });
    } else if (tableName == "TaskSchedulerDevice") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::TaskSchedulerDevice>(
            HcclSim::Storage::All<sim::runtime::TaskSchedulerDevice>());
        std::vector<sim::runtime::TaskSchedulerDevice> tables
            = selected.value_or(std::vector<sim::runtime::TaskSchedulerDevice>{});
        PrintTable<sim::runtime::TaskSchedulerDevice>(
            "| id | device_id | type|", tables, [](const sim::runtime::TaskSchedulerDevice& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.type) + " |";
            });
    } else if (tableName == "ComputeDie") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::ComputeDie>(HcclSim::Storage::All<sim::runtime::ComputeDie>());
        std::vector<sim::runtime::ComputeDie> tables = selected.value_or(std::vector<sim::runtime::ComputeDie>{});
        PrintTable<sim::runtime::ComputeDie>("| id | ts_id | type |", tables, [](const sim::runtime::ComputeDie& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ts_id) + " | " + std::to_string(tmp.type)
                   + " |";
        });
    } else if (tableName == "DeviceStatus") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::DeviceStatus>(
            HcclSim::Storage::All<sim::runtime::DeviceStatus>());
        std::vector<sim::runtime::DeviceStatus> tables = selected.value_or(std::vector<sim::runtime::DeviceStatus>{});
        PrintTable<sim::runtime::DeviceStatus>(
            "| id | device_id | overflow | sync_strat | sync_timeout | capability | run_by_host | ts_core | online |",
            tables, [](const sim::runtime::DeviceStatus& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.overflow_status) + " | " + std::to_string(tmp.synchronize_strategy) + " | "
                       + std::to_string(tmp.synchronize_timeout) + " | " + std::to_string(tmp.capability_mask) + " | "
                       + std::to_string(tmp.run_by_host) + " | " + std::to_string(tmp.ts_core) + " | "
                       + std::to_string(tmp.online_status) + " |";
            });
    } else if (tableName == "Port") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::Port>(HcclSim::Storage::All<sim::runtime::Port>());
        std::vector<sim::runtime::Port> tables = selected.value_or(std::vector<sim::runtime::Port>{});
        PrintTable<sim::runtime::Port>(
            "| id | device_id | die_id | name | status |", tables, [](const sim::runtime::Port& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.die_id) + " | " + std::string(tmp.name) + " | " + std::to_string(tmp.status)
                       + "|";
            });
    } else if (tableName == "Ccu") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::Ccu>(HcclSim::Storage::All<sim::runtime::Ccu>());
        std::vector<sim::runtime::Ccu> tables = selected.value_or(std::vector<sim::runtime::Ccu>{});
        PrintTable<sim::runtime::Ccu>(
            "| id | device_id | resource_addr | die_id | status |", tables, [](const sim::runtime::Ccu& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.resource_addr) + " | " + std::to_string(tmp.die_id) + " | "
                       + std::to_string(tmp.status) + " |";
            });
    } else if (tableName == "DeviceConnection") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::DeviceConnection>(
            HcclSim::Storage::All<sim::runtime::DeviceConnection>());
        std::vector<sim::runtime::DeviceConnection> tables
            = selected.value_or(std::vector<sim::runtime::DeviceConnection>{});
        PrintTable<sim::runtime::DeviceConnection>(
            "| id | src_dev_id | dst_dev_id | link_type | access_by_remote |", tables,
            [](const sim::runtime::DeviceConnection& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.src_dev_id) + " | "
                       + std::to_string(tmp.dst_dev_id) + " | " + std::to_string(tmp.link_type) + " | "
                       + std::to_string(tmp.access_by_remote) + " |";
            });
    } else if (tableName == "EndPoint") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::EndPoint>(HcclSim::Storage::All<sim::runtime::EndPoint>());
        std::vector<sim::runtime::EndPoint> tables = selected.value_or(std::vector<sim::runtime::EndPoint>{});
        PrintTable<sim::runtime::EndPoint>(
            "| id | device_id | func_id | die_id | type | eid | ip_addr | is_uboe |", tables,
            [](const sim::runtime::EndPoint& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.func_id) + " | " + std::to_string(tmp.die_id) + " | "
                       + std::to_string(tmp.type) + " | " + std::string(reinterpret_cast<const char*>(tmp.eid)) + " | "
                       + std::string(tmp.ip_addr) + " | " + std::to_string(tmp.is_uboe) + " |";
            });
    } else if (tableName == "EndPointPair") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::EndPointPair>(
            HcclSim::Storage::All<sim::runtime::EndPointPair>());
        std::vector<sim::runtime::EndPointPair> tables = selected.value_or(std::vector<sim::runtime::EndPointPair>{});
        PrintTable<sim::runtime::EndPointPair>(
            "| id | local_enpoint_id | remote_enpoint_id | tp_type |", tables,
            [](const sim::runtime::EndPointPair& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.local_enpoint_id) + " | "
                       + std::to_string(tmp.remote_enpoint_id) + " | " + std::to_string(tmp.tp_type) + " |";
            });
    } else if (tableName == "EndPointPortMapping") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::EndPointPortMapping>(
            HcclSim::Storage::All<sim::runtime::EndPointPortMapping>());
        std::vector<sim::runtime::EndPointPortMapping> tables
            = selected.value_or(std::vector<sim::runtime::EndPointPortMapping>{});
        PrintTable<sim::runtime::EndPointPortMapping>(
            "| id | port_id | endpoint_id | net_layer |", tables, [](const sim::runtime::EndPointPortMapping& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.port_id) + " | "
                       + std::to_string(tmp.endpoint_id) + " | " + std::to_string(tmp.net_layer) + " |";
            });
    } else if (tableName == "Link") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::Link>(HcclSim::Storage::All<sim::runtime::Link>());
        std::vector<sim::runtime::Link> tables = selected.value_or(std::vector<sim::runtime::Link>{});
        PrintTable<sim::runtime::Link>(
            "| id | local_endpoint_id | remote_endpoint_id | net_layer | type | protocols |", tables,
            [](const sim::runtime::Link& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.local_endpoint_id) + " | "
                       + std::to_string(tmp.remote_endpoint_id) + " | " + std::to_string(tmp.net_layer) + " | "
                       + std::to_string(tmp.type) + " | " + std::to_string(tmp.protocols[0]) + " |";
            });
    } else if (tableName == "CcuChannel") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::CcuChannel>(HcclSim::Storage::All<sim::runtime::CcuChannel>());
        std::vector<sim::runtime::CcuChannel> tables = selected.value_or(std::vector<sim::runtime::CcuChannel>{});
        PrintTable<sim::runtime::CcuChannel>(
            "| id | channel_id | local_endpoint_id | remote_endpoint_id | protocol | jetty_start | jetty_num |", tables,
            [](const sim::runtime::CcuChannel& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.channel_id) + " | "
                       + std::to_string(tmp.local_endpoint_id) + " | " + std::to_string(tmp.remote_endpoint_id) + " | "
                       + std::to_string(tmp.protocol) + " | " + std::to_string(tmp.jetty_start) + " | "
                       + std::to_string(tmp.jetty_num) + " |";
            });
    } else if (tableName == "Context") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::Context>(HcclSim::Storage::All<sim::runtime::Context>());
        std::vector<sim::runtime::Context> tables = selected.value_or(std::vector<sim::runtime::Context>{});
        PrintTable<sim::runtime::Context>(
            "| id | run_id | device_id | is_default | ref_cnt | float_overflow_addr | capture_mode |", tables,
            [](const sim::runtime::Context& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.run_id) + " | "
                       + std::to_string(tmp.device_id) + " | " + std::to_string(tmp.is_default) + " | "
                       + std::to_string(tmp.ref_cnt) + " | " + std::to_string(tmp.float_overflow_addr) + " | "
                       + std::to_string(tmp.capture_mode) + " |";
            });
    } else if (tableName == "Stream") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::Stream>(HcclSim::Storage::All<sim::runtime::Stream>());
        std::vector<sim::runtime::Stream> tables = selected.value_or(std::vector<sim::runtime::Stream>{});
        PrintTable<sim::runtime::Stream>(
            "| id | ctx_id | sq_base_addr | is_primary_default | is_other_default | priority | schedule_strategy | "
            "failure_mode | user_tag | overflow_switch | activated | capture_status | task_complete_status |",
            tables, [](const sim::runtime::Stream& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_id) + " | "
                       + std::to_string(tmp.sq_base_addr) + " | " + std::to_string(tmp.is_primary_default) + " | "
                       + std::to_string(tmp.is_other_default) + " | " + std::to_string(tmp.priority) + " | "
                       + std::to_string(tmp.schedule_strategy) + " | " + std::to_string(tmp.failure_mode) + " | "
                       + std::to_string(tmp.user_tag) + " | " + std::to_string(tmp.overflow_switch) + " | "
                       + std::to_string(tmp.activated) + " | " + std::to_string(tmp.capture_status) + " | "
                       + std::to_string(tmp.task_complete_status) + " |";
            });
    } else if (tableName == "Task") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::Task>(HcclSim::Storage::All<sim::runtime::Task>());
        std::vector<sim::runtime::Task> tables = selected.value_or(std::vector<sim::runtime::Task>{});
        PrintTable<sim::runtime::Task>(
            "| id | stream_id | cid | seq_number | type |", tables, [](const sim::runtime::Task& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.stream_id) + " | "
                       + std::to_string(tmp.cid) + " | " + std::to_string(tmp.seq_number) + " | "
                       + std::to_string(tmp.type) + " |";
            });
    } else if (tableName == "EventSyncTask") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::EventSyncTask>(
            HcclSim::Storage::All<sim::runtime::EventSyncTask>());
        std::vector<sim::runtime::EventSyncTask> tables = selected.value_or(std::vector<sim::runtime::EventSyncTask>{});
        PrintTable<sim::runtime::EventSyncTask>(
            "| id | execute_time_ms | finish_time_ms | op_timeout_s |", tables,
            [](const sim::runtime::EventSyncTask& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.execute_time_ms) + " | "
                       + std::to_string(tmp.finish_time_ms) + " | " + std::to_string(tmp.op_timeout_s) + " |";
            });
    } else if (tableName == "Notify") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::Notify>(HcclSim::Storage::All<sim::runtime::Notify>());
        std::vector<sim::runtime::Notify> tables = selected.value_or(std::vector<sim::runtime::Notify>{});
        PrintTable<sim::runtime::Notify>(
            "| id | create_ctx_id | device_notify_seq | value |", tables, [](const sim::runtime::Notify& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.create_ctx_id) + " | "
                       + std::to_string(tmp.device_notify_seq) + " | " + std::to_string(tmp.value) + " |";
            });
    } else if (tableName == "IpcNotify") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::IpcNotify>(HcclSim::Storage::All<sim::runtime::IpcNotify>());
        std::vector<sim::runtime::IpcNotify> tables = selected.value_or(std::vector<sim::runtime::IpcNotify>{});
        PrintTable<sim::runtime::IpcNotify>(
            " id | notify_id | name_or_key | create_pid |", tables, [](const sim::runtime::IpcNotify& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.notify_id) + " | "
                       + std::string(reinterpret_cast<const char*>(tmp.name_or_key)) + " | "
                       + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "IpcNotifyVistorList") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::IpcNotifyVistorList>(
            HcclSim::Storage::All<sim::runtime::IpcNotifyVistorList>());
        std::vector<sim::runtime::IpcNotifyVistorList> tables
            = selected.value_or(std::vector<sim::runtime::IpcNotifyVistorList>{});
        PrintTable<sim::runtime::IpcNotifyVistorList>(
            "| ipc_id | visitor_pid |", tables, [](const sim::runtime::IpcNotifyVistorList& tmp) {
                return "| " + std::to_string(tmp.ipc_id) + " | " + std::to_string(tmp.visitor_pid) + " |";
            });
    } else if (tableName == "NotifyRecordTask") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::NotifyRecordTask>(
            HcclSim::Storage::All<sim::runtime::NotifyRecordTask>());
        std::vector<sim::runtime::NotifyRecordTask> tables
            = selected.value_or(std::vector<sim::runtime::NotifyRecordTask>{});
        PrintTable<sim::runtime::NotifyRecordTask>(
            "| notify_id |", tables, [](const sim::runtime::NotifyRecordTask& tmp) {
                return "| " + std::to_string(tmp.notify_id) + " |";
            });
    } else if (tableName == "NotifyWaitTask") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::NotifyWaitTask>(
            HcclSim::Storage::All<sim::runtime::NotifyWaitTask>());
        std::vector<sim::runtime::NotifyWaitTask> tables
            = selected.value_or(std::vector<sim::runtime::NotifyWaitTask>{});
        PrintTable<sim::runtime::NotifyWaitTask>("| notify_id |", tables, [](const sim::runtime::NotifyWaitTask& tmp) {
            return "| " + std::to_string(tmp.notify_id) + " |";
        });
    } else if (tableName == "Event") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::Event>(HcclSim::Storage::All<sim::runtime::Event>());
        std::vector<sim::runtime::Event> tables = selected.value_or(std::vector<sim::runtime::Event>{});
        PrintTable<sim::runtime::Event>(
            "| id | create_ctx_id | event_flag | device_res_seq | created_time | status |", tables,
            [](const sim::runtime::Event& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.create_ctx_id) + " | "
                       + std::to_string(tmp.event_flag) + " | " + std::to_string(tmp.device_res_seq) + " | "
                       + std::to_string(tmp.created_time) + " | " + std::to_string(tmp.status) + " |";
            });
    } else if (tableName == "PhyMem") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::PhyMemBlock>(
            HcclSim::Storage::All<sim::runtime::PhyMemBlock>());
        std::vector<sim::runtime::PhyMemBlock> tables = selected.value_or(std::vector<sim::runtime::PhyMemBlock>{});
        PrintTable<sim::runtime::PhyMemBlock>(
            "| id | device_id | name | size | type | ref_count |", tables, [](const sim::runtime::PhyMemBlock& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::string(tmp.name) + " | " + std::to_string(tmp.size) + " | " + std::to_string(tmp.type)
                       + " | " + std::to_string(tmp.ref_count) + " |";
            });
    } else if (tableName == "VirMem") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::VirtualMemBlock>(
            HcclSim::Storage::All<sim::runtime::VirtualMemBlock>());
        std::vector<sim::runtime::VirtualMemBlock> tables
            = selected.value_or(std::vector<sim::runtime::VirtualMemBlock>{});
        PrintTable<sim::runtime::VirtualMemBlock>(
            "| id | start_ptr | size | ctx_id | phy_mem_id | owner_pid | src_type | policy |", tables,
            [](const sim::runtime::VirtualMemBlock& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.start_ptr) + " | "
                       + std::to_string(tmp.size) + " | " + std::to_string(tmp.ctx_id) + " | "
                       + std::to_string(tmp.phy_mem_id) + " | " + std::to_string(tmp.owner_pid) + " | "
                       + std::to_string(tmp.src_type) + " | " + std::to_string(tmp.policy) + " |";
            });
    } else if (tableName == "IpcMemRecord") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::IpcMemRecord>(
            HcclSim::Storage::All<sim::runtime::IpcMemRecord>());
        std::vector<sim::runtime::IpcMemRecord> tables = selected.value_or(std::vector<sim::runtime::IpcMemRecord>{});
        PrintTable<sim::runtime::IpcMemRecord>(
            "| id | vir_mem_id | offset | create_pid |", tables, [](const sim::runtime::IpcMemRecord& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.vir_mem_id) + " | "
                       + std::to_string(tmp.offset) + " | " + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "IpcMemWhiteList") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::IpcMemWhiteList>(
            HcclSim::Storage::All<sim::runtime::IpcMemWhiteList>());
        std::vector<sim::runtime::IpcMemWhiteList> tables
            = selected.value_or(std::vector<sim::runtime::IpcMemWhiteList>{});
        PrintTable<sim::runtime::IpcMemWhiteList>(
            "| id | name_or_key | pid | create_pid |", tables, [](const sim::runtime::IpcMemWhiteList& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.name_or_key) + " | "
                       + std::to_string(tmp.pid) + " | " + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "FdMemRecord") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::FdMemRecord>(
            HcclSim::Storage::All<sim::runtime::FdMemRecord>());
        std::vector<sim::runtime::FdMemRecord> tables = selected.value_or(std::vector<sim::runtime::FdMemRecord>{});
        PrintTable<sim::runtime::FdMemRecord>(
            "| id | vir_mem_id | phy_mem_id | create_pid |", tables, [](const sim::runtime::FdMemRecord& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.vir_mem_id) + " | "
                       + std::to_string(tmp.phy_mem_id) + " | " + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "FdMemWhiteList") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::FdMemWhiteList>(
            HcclSim::Storage::All<sim::runtime::FdMemWhiteList>());
        std::vector<sim::runtime::FdMemWhiteList> tables
            = selected.value_or(std::vector<sim::runtime::FdMemWhiteList>{});
        PrintTable<sim::runtime::FdMemWhiteList>(
            "| id | name_or_key | pid | create_pid |", tables, [](const sim::runtime::FdMemWhiteList& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.name_or_key) + " | "
                       + std::to_string(tmp.pid) + " | " + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "RaSocketPair") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::RaSocketPair>(
            HcclSim::Storage::All<sim::runtime::RaSocketPair>());
        std::vector<sim::runtime::RaSocketPair> tables = selected.value_or(std::vector<sim::runtime::RaSocketPair>{});
        PrintTable<sim::runtime::RaSocketPair>(
            "| id | server_id | client_id | ref_cnt | port | tag_hash |", tables,
            [](const sim::runtime::RaSocketPair& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.server_id) + " | "
                       + std::to_string(tmp.client_id) + " | " + std::to_string(tmp.ref_cnt) + " | "
                       + std::to_string(tmp.port) + " | " + std::to_string(tmp.tag_hash) + " |";
            });
    } else if (tableName == "MemoryLayout") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::MemoryLayout>(
            HcclSim::Storage::All<sim::runtime::MemoryLayout>());
        std::vector<sim::runtime::MemoryLayout> tables = selected.value_or(std::vector<sim::runtime::MemoryLayout>{});
        PrintTable<sim::runtime::MemoryLayout>(
            "| id | rank_id | base_addr | buf_type | reserved | size | global_offset |", tables,
            [](const sim::runtime::MemoryLayout& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.rank_id) + " | "
                       + std::to_string(tmp.base_addr) + " | " + std::to_string(tmp.buf_type) + " | "
                       + std::to_string(tmp.reserved) + " | " + std::to_string(tmp.size) + " | "
                       + std::to_string(tmp.global_offset) + " |";
            });
    } else if (tableName == "SimModelData") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::SimModelData>(
            HcclSim::Storage::All<sim::runtime::SimModelData>());
        std::vector<sim::runtime::SimModelData> tables = selected.value_or(std::vector<sim::runtime::SimModelData>{});
        PrintTable<sim::runtime::SimModelData>(
            "| id | rank_id | src_rank | dst_rank | root | rank_size | chip_type | op_type | reduce_op | data_type | "
            "data_count | op_expansion_mode | ccu0_resource_base_addr | ccu1_resource_base_addr |",
            tables, [](const sim::runtime::SimModelData& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.rank_id) + " | "
                       + std::to_string(tmp.src_rank) + " | " + std::to_string(tmp.dst_rank) + " | "
                       + std::to_string(tmp.root) + " | " + std::to_string(tmp.rank_size) + " | "
                       + std::to_string(tmp.chip_type) + " | " + std::to_string(tmp.op_type) + " | "
                       + std::to_string(tmp.reduce_op) + " | " + std::to_string(tmp.data_type) + " | "
                       + std::to_string(tmp.data_count) + " | " + std::to_string(tmp.op_expansion_mode) + " | "
                       + std::to_string(tmp.ccu0_resource_base_addr) + " | "
                       + std::to_string(tmp.ccu1_resource_base_addr) + " |";
            });
    } else if (tableName == "DpuDeviceInfo") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::DpuDeviceInfo>(
            HcclSim::Storage::All<sim::runtime::DpuDeviceInfo>());
        std::vector<sim::runtime::DpuDeviceInfo> tables = selected.value_or(std::vector<sim::runtime::DpuDeviceInfo>{});
        PrintTable<sim::runtime::DpuDeviceInfo>(
            "| id | device_id | stream_id |", tables, [](const sim::runtime::DpuDeviceInfo& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.stream_id) + " |";
            });
    } else if (tableName == "RaDevice") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::RaDevice>(HcclSim::Storage::All<sim::runtime::RaDevice>());
        std::vector<sim::runtime::RaDevice> tables = selected.value_or(std::vector<sim::runtime::RaDevice>{});
        PrintTable<sim::runtime::RaDevice>(
            "| id | device_id | mac_addr | state | endpoint_id |", tables, [](const sim::runtime::RaDevice& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::string((char*)tmp.mac_addr) + " | " + std::to_string(tmp.state) + " |"
                       + std::to_string(tmp.endpoint_id) + " |";
            });
    } else if (tableName == "RaQP") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::RaQP>(HcclSim::Storage::All<sim::runtime::RaQP>());
        std::vector<sim::runtime::RaQP> tables = selected.value_or(std::vector<sim::runtime::RaQP>{});
        PrintTable<sim::runtime::RaQP>(
            "| id | ra_dev_id | send_cq_handle | recv_cq_handle | qp_num | type | state | taJettyId | mode | "
            "peer_qp_id | peer_qpn | perr_lid | pid |",
            tables, [](const sim::runtime::RaQP& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ra_dev_id) + " | "
                       + std::to_string(tmp.send_cq_handle) + " | " + std::to_string(tmp.recv_cq_handle) + " | "
                       + std::to_string(tmp.qp_num) + " | " + std::to_string(tmp.type) + " | "
                       + std::to_string(tmp.state) + " | " + std::to_string(tmp.taJettyId) + " | "
                       + std::to_string(tmp.mode) + " | " + std::to_string(tmp.peer_qp_id) + " | "
                       + std::to_string(tmp.peer_qpn) + " | " + std::to_string(tmp.perr_lid) + " | "
                       + std::to_string(tmp.pid) + " |";
            });
    } else if (tableName == "RaCQ") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::RaCQ>(HcclSim::Storage::All<sim::runtime::RaCQ>());
        std::vector<sim::runtime::RaCQ> tables = selected.value_or(std::vector<sim::runtime::RaCQ>{});
        PrintTable<sim::runtime::RaCQ>("| id | ra_dev_id | cqn | size |", tables, [](const sim::runtime::RaCQ& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ra_dev_id) + " | "
                   + std::to_string(tmp.cqn) + " | " + std::to_string(tmp.size) + " |";
        });
    } else if (tableName == "RaCQE") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::RaCQE>(HcclSim::Storage::All<sim::runtime::RaCQE>());
        std::vector<sim::runtime::RaCQE> tables = selected.value_or(std::vector<sim::runtime::RaCQE>{});
        PrintTable<sim::runtime::RaCQE>(
            "| id | cq_handle | wr_id | status |", tables, [](const sim::runtime::RaCQE& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.cq_handle) + " | "
                       + std::to_string(tmp.wr_id) + " | " + std::to_string(tmp.status) + " |";
            });
    } else if (tableName == "RaMR") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::RaMR>(HcclSim::Storage::All<sim::runtime::RaMR>());
        std::vector<sim::runtime::RaMR> tables = selected.value_or(std::vector<sim::runtime::RaMR>{});
        PrintTable<sim::runtime::RaMR>(
            "| id | local_key | remote_key | vptr_id | length | addr |", tables, [](const sim::runtime::RaMR& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.local_key) + " | "
                       + std::to_string(tmp.remote_key) + " | " + std::to_string(tmp.vptr_id) + " | "
                       + std::to_string(tmp.length) + " | " + std::to_string(tmp.addr) + " |";
            });
    } else if (tableName == "RaContext") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::RaContext>(HcclSim::Storage::All<sim::runtime::RaContext>());
        std::vector<sim::runtime::RaContext> tables = selected.value_or(std::vector<sim::runtime::RaContext>{});
        PrintTable<sim::runtime::RaContext>(
            "| id | device_id | mode | endpoint_id | eidIndex | max_jetty_num | max_jfc_num |", tables,
            [](const sim::runtime::RaContext& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.mode) + " | " + std::to_string(tmp.endpoint_id) + " | "
                       + std::to_string(tmp.eidIndex) + " | " + std::to_string(tmp.max_jetty_num) + " | "
                       + std::to_string(tmp.max_jfc_num) + " |";
            });
    } else if (tableName == "RaChan") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::RaChan>(HcclSim::Storage::All<sim::runtime::RaChan>());
        std::vector<sim::runtime::RaChan> tables = selected.value_or(std::vector<sim::runtime::RaChan>{});
        PrintTable<sim::runtime::RaChan>(
            "| id | ctx_handle | chann_id | mode |", tables, [](const sim::runtime::RaChan& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                       + std::to_string(tmp.chann_id) + " | " + std::to_string(tmp.mode) + " |";
            });
    } else if (tableName == "RaTokenId") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::RaTokenId>(HcclSim::Storage::All<sim::runtime::RaTokenId>());
        std::vector<sim::runtime::RaTokenId> tables = selected.value_or(std::vector<sim::runtime::RaTokenId>{});
        PrintTable<sim::runtime::RaTokenId>(
            "| id | ctx_handle | token_id | ref_count |", tables, [](const sim::runtime::RaTokenId& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                       + std::to_string(tmp.token_id) + " | " + std::to_string(tmp.ref_count) + " |";
            });
    } else if (tableName == "RaTp") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::RaTp>(HcclSim::Storage::All<sim::runtime::RaTp>());
        std::vector<sim::runtime::RaTp> tables = selected.value_or(std::vector<sim::runtime::RaTp>{});
        PrintTable<sim::runtime::RaTp>(
            "| id | ctx_handle | tp_type | tpn | speed | status |", tables, [](const sim::runtime::RaTp& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                       + std::to_string(tmp.tp_type) + " | " + std::to_string(tmp.tpn) + " | "
                       + std::to_string(tmp.speed) + " | " + std::to_string(tmp.status) + " |";
            });
    } else if (tableName == "RaRmem") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::RaRmem>(HcclSim::Storage::All<sim::runtime::RaRmem>());
        std::vector<sim::runtime::RaRmem> tables = selected.value_or(std::vector<sim::runtime::RaRmem>{});
        PrintTable<sim::runtime::RaRmem>(
            "| id | ctx_handle | remote_key | target_seg_handle | remote_eid |", tables,
            [](const sim::runtime::RaRmem& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                       + std::to_string(tmp.remote_key) + " | " + std::to_string(tmp.target_seg_handle) + " | "
                       + std::to_string(tmp.remote_eid) + " |";
            });
    } else if (tableName == "RaLmem") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::RaLmem>(HcclSim::Storage::All<sim::runtime::RaLmem>());
        std::vector<sim::runtime::RaLmem> tables = selected.value_or(std::vector<sim::runtime::RaLmem>{});
        PrintTable<sim::runtime::RaLmem>(
            "| id | ctx_handle | addr | size | mem_key | token_id |", tables, [](const sim::runtime::RaLmem& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                       + std::to_string(tmp.addr) + " | " + std::to_string(tmp.size) + " | "
                       + std::to_string(tmp.mem_key) + " | " + std::to_string(tmp.token_id) + " |";
            });
    } else if (tableName == "RaJetty") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::RaJetty>(HcclSim::Storage::All<sim::runtime::RaJetty>());
        std::vector<sim::runtime::RaJetty> tables = selected.value_or(std::vector<sim::runtime::RaJetty>{});
        PrintTable<sim::runtime::RaJetty>(
            "| id | ctx_handle | send_cq_handle | recv_cq_handle | sqDepth | rqDepth | type | jetty_id | dieId | "
            "rank_id | state | mode | peer_jetty_handle | peer_endpoint_id | pid |",
            tables, [](const sim::runtime::RaJetty& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                       + std::to_string(tmp.send_cq_handle) + " | " + std::to_string(tmp.recv_cq_handle) + " | "
                       + std::to_string(tmp.sqDepth) + " | " + std::to_string(tmp.rqDepth) + " | "
                       + std::to_string(tmp.type) + " | " + std::to_string(tmp.jetty_id) + " | "
                       + std::to_string(tmp.dieId) + " | " + std::to_string(tmp.rank_id) + " | "
                       + std::to_string(tmp.state) + " | " + std::to_string(tmp.mode) + " | "
                       + std::to_string(tmp.peer_jetty_handle) + " | " + std::to_string(tmp.peer_endpoint_id) + " | "
                       + std::to_string(tmp.pid) + " |";
            });
    } else if (tableName == "RaJfc") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::RaJfc>(HcclSim::Storage::All<sim::runtime::RaJfc>());
        std::vector<sim::runtime::RaJfc> tables = selected.value_or(std::vector<sim::runtime::RaJfc>{});
        PrintTable<sim::runtime::RaJfc>(
            "| id | ctx_handle | jfc_id | depth | mode | policy |", tables, [](const sim::runtime::RaJfc& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                       + std::to_string(tmp.jfc_id) + " | " + std::to_string(tmp.depth) + " | "
                       + std::to_string(tmp.mode) + " | " + std::to_string(tmp.policy) + " |";
            });
    } else if (tableName == "RaCr") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::RaCr>(HcclSim::Storage::All<sim::runtime::RaCr>());
        std::vector<sim::runtime::RaCr> tables = selected.value_or(std::vector<sim::runtime::RaCr>{});
        PrintTable<sim::runtime::RaCr>(
            "| id | jfc_handle | user_ctx | status | opcode | byte_len |", tables, [](const sim::runtime::RaCr& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.jfc_handle) + " | "
                       + std::to_string(tmp.user_ctx) + " | " + std::to_string(tmp.status) + " | "
                       + std::to_string(tmp.opcode) + " | " + std::to_string(tmp.byte_len) + " |";
            });
    } else if (tableName == "RaTlv") {
        auto selected = sim::runtime::Db::GetByPred<sim::runtime::RaTlv>(HcclSim::Storage::All<sim::runtime::RaTlv>());
        std::vector<sim::runtime::RaTlv> tables = selected.value_or(std::vector<sim::runtime::RaTlv>{});
        PrintTable<sim::runtime::RaTlv>("| id | physical_id |", tables, [](const sim::runtime::RaTlv& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.physical_id) + " |";
        });
    } else if (tableName == "Plugin") {
        auto selected
            = sim::runtime::Db::GetByPred<sim::runtime::Plugin>(HcclSim::Storage::All<sim::runtime::Plugin>());
        std::vector<sim::runtime::Plugin> tables = selected.value_or(std::vector<sim::runtime::Plugin>{});
        PrintTable<sim::runtime::Plugin>("| id | tag |", tables, [](const sim::runtime::Plugin& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::string(tmp.tag) + " |";
        });
    } else {
        HCCL_VM_ERROR("undefine table {}", tableName);
    }
}

bool CmdTableUpdate(const std::string& table, const uint64_t id, const std::string& column, const std::string& value)
{
    if (table == "Device" && column == "soc_version") {
        // 显式字段更新:soc_version 是 TEXT 列,新值按 C 字符串语义写入。
        auto updated = sim::runtime::Db::Update<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::id, id),
            HcclSim::Storage::Set(&sim::runtime::Device::soc_version, value));
        return updated.ok() && updated.value.has_value() && *updated.value > 0;
    } else {
        HCCL_VM_ERROR("undefine update {} [id={}].{} = \"{}\"", table, id, column, value);
        return false;
    }
}
} // namespace HcclSim
