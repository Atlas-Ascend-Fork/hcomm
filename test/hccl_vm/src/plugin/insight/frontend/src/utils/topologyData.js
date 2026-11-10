/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

const DEFAULT_API_ROOT = ''

function joinApiPath(apiRoot, path) {
  const normalizedRoot = apiRoot.endsWith('/') ? apiRoot.slice(0, -1) : apiRoot
  return `${normalizedRoot}${path}`
}

async function requestJson(path, { apiRoot = DEFAULT_API_ROOT } = {}) {
  const response = await fetch(joinApiPath(apiRoot, path))
  const text = await response.text()
  const payload = text ? JSON.parse(text) : {}
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`)
  }
  return payload
}

export async function fetchTopologyList(options) {
  const payload = await requestJson('/api/topology/list', options)
  return payload.topologies || []
}

export async function fetchClusterTopology(topoName, options) {
  return requestJson(`/api/topology/cluster/${encodeURIComponent(topoName)}`, options)
}

export function parseClusterTopology(raw) {
  const superPods = []
  const spEntries = raw.super_pods || {}

  for (const [spIdStr, servers] of Object.entries(spEntries)) {
    const spId = Number(spIdStr)
    const parsedServers = (servers || []).map((srv) => ({
      id: srv.id,
      serverTopo: srv.server_topo || '',
      socVersion: srv.soc_version || '',
    }))

    const serverTopoKey = parsedServers.length > 0 ? parsedServers[0].serverTopo : ''
    const serverTopoConfig = (raw.server_topo_configs || {})[serverTopoKey] || null

    superPods.push({
      id: spId,
      servers: parsedServers,
      serverTopoConfig,
    })
  }

  superPods.sort((a, b) => a.id - b.id)

  return {
    name: raw.name || '',
    description: raw.description || '',
    superNodeNum: raw.super_node_num || 0,
    serverNum: raw.server_num || 0,
    isNewFormat: raw.is_new_format || false,
    superPods,
  }
}

export function parseServerTopoConfig(config) {
  if (!config) return null

  const deviceNum = config.device_num || 0
  const socVersion = config.soc_version || ''
  const hardwareType = config.hardware_type || ''

  // POD形态(board_num > 1): links中的device id为board内物理id, 渲染前需换算为
  // 全局device id (= board_id * 每board卡数 + board内物理id); server形态换算为恒等映射
  const boardNum = Number(config.board_num) > 1 ? Number(config.board_num) : 1
  const nodeDeviceNum = boardNum > 1 && deviceNum % boardNum === 0 ? deviceNum / boardNum : deviceNum

  const pinMapEntries = (config.device_ports_allocate_map || []).map((entry) => ({
    dieId: entry.die_id,
    pinMap: entry.pin_map || [],
  }))

  const portGroups = (config.port_group || []).map((pg) => ({
    layer: pg.layer,
    ports: pg.ports || [],
  }))

  const links = parseLinks(config.links || [], boardNum, nodeDeviceNum)

  return {
    deviceNum,
    socVersion,
    hardwareType,
    pinMapEntries,
    portGroups,
    links,
  }
}

// 展开一个条目的device id列表: 支持 `name`(标量/列表) 与 `name_range`([start,end]闭区间或列表)
function expandIdField(entry, name) {
  const ids = []
  const raw = entry[name]
  if (Array.isArray(raw)) {
    ids.push(...raw.map(Number))
  } else if (raw !== undefined && raw !== null) {
    ids.push(Number(raw))
  }
  const range = entry[`${name}_range`]
  if (Array.isArray(range) && range.length === 2) {
    const start = Number(range[0])
    const end = Number(range[1])
    for (let i = start; i <= end; i++) ids.push(i)
  } else if (Array.isArray(range)) {
    ids.push(...range.map(Number))
  }
  return ids
}

// 条目适用的board列表: 配置board_id时仅该board; 未配置时对所有board生效(单board描述复制扩展)
function boardsForEntry(entry, boardNum) {
  if (entry.board_id !== undefined && entry.board_id !== null) return [Number(entry.board_id)]
  return Array.from({ length: boardNum }, (_, b) => b)
}

// links解析: device id统一换算为全局device id渲染。
//   - fullmesh的connections/device_to_switch_links: 单个board_id字段声明所属board;
//   - device_to_device_links: src_board_id/dst_board_id声明两端所属board(支持跨board直连),
//     device id字段为新命名src_device_id(_range)/dst_device_id(_range), 兼容旧命名src_local_id(_range)/dst_local_id(_range);
//   - server形态(boardNum=1)换算为恒等映射, 行为与既有渲染一致。
function parseLinks(rawLinks, boardNum = 1, nodeDeviceNum = 0) {
  const result = { fullmesh: [], enum: { deviceToDevice: [], deviceToSwitch: [] } }
  const toGlobal = (boardId, localId) => boardId * nodeDeviceNum + Number(localId)

  for (const link of rawLinks) {
    const mode = link.link_mode || ''
    if (mode === 'fullmesh') {
      for (const c of link.connections || []) {
        const dieId = c.die_id
        let locals = expandIdField(c, 'devices')
        if (locals.length === 0) locals = expandIdField(c, 'device_id')
        for (const b of boardsForEntry(c, boardNum)) {
          result.fullmesh.push({ dieId, devices: locals.map((id) => toGlobal(b, id)) })
        }
      }
    } else if (mode === 'enum') {
      for (const c of link.device_to_device_links || []) {
        const srcDieId = c.src_die_id !== undefined ? c.src_die_id : c.die_id
        const dstDieId = c.dst_die_id !== undefined ? c.dst_die_id : srcDieId
        let srcLocal = expandIdField(c, 'src_device_id')
        if (srcLocal.length === 0) srcLocal = expandIdField(c, 'src_local_id')
        let dstLocal = expandIdField(c, 'dst_device_id')
        if (dstLocal.length === 0) dstLocal = expandIdField(c, 'dst_local_id')
        const hasSrcBoard = c.src_board_id !== undefined && c.src_board_id !== null
        const hasDstBoard = c.dst_board_id !== undefined && c.dst_board_id !== null
        const pairs = (hasSrcBoard || hasDstBoard)
          ? [[Number(c.src_board_id), Number(c.dst_board_id)]]
          : boardsForEntry(c, boardNum).map((b) => [b, b])
        for (const [sb, db] of pairs) {
          result.enum.deviceToDevice.push({
            srcDieId,
            dstDieId,
            srcLocalId: srcLocal.map((id) => toGlobal(sb, id)),
            dstLocalId: dstLocal.map((id) => toGlobal(db, id)),
            srcLocalIdRange: null,
            dstLocalIdRange: null,
          })
        }
      }
      for (const c of link.device_to_switch_links || []) {
        const dieId = c.die_id
        const locals = expandIdField(c, 'devices')
        for (const b of boardsForEntry(c, boardNum)) {
          result.enum.deviceToSwitch.push({ dieId, devices: locals.map((id) => toGlobal(b, id)) })
        }
      }
    }
  }

  return result
}

export function getPinMapType(pinMap) {
  const typeMap = { 0: 'unused', 1: 'p2p', 2: 'p2net', 3: 'd2h' }
  return pinMap.map((v) => typeMap[v] || 'unknown')
}

export function getPortGroupType(portGroup, pinMapEntries) {
  if (!portGroup.ports || portGroup.ports.length === 0) return 'unknown'
  const firstPort = portGroup.ports[0]
  const parts = firstPort.split('/')
  if (parts.length < 2) return 'unknown'
  const dieId = Number(parts[0])
  const portId = Number(parts[1])
  const entry = pinMapEntries.find((e) => e.dieId === dieId)
  if (!entry || portId >= entry.pinMap.length) return 'unknown'
  const typeMap = { 0: 'unused', 1: 'p2p', 2: 'p2net', 3: 'd2h' }
  return typeMap[entry.pinMap[portId]] || 'unknown'
}
