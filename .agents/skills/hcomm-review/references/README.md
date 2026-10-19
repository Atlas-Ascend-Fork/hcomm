# 检视规范索引

检视规范按七类组织，检视时按需加载对应文档（非流程；流程见 [`SKILL.md`](../SKILL.md)）。仓内编码/架构规范演进时以根 `AGENTS.md` 与其链接的权威文档为准，本文档体系与之保持同步。

| 文档 | 覆盖 | 何时加载 |
|------|------|---------|
| [`coding-and-security.md`](./coding-and-security.md) | 命名风格、内存/资源/并发/错误处理红线、工具验证纪律 | 所有代码 PR 必读 |
| [`external-api.md`](./external-api.md) | 对外头文件（include/、pkg_inc/）C 接口、ABI、命名一致性、兼容性、模块变更 | PR 触碰 include/、pkg_inc/ 或新增类/文件/目录时 |
| [`architecture.md`](./architecture.md) | 分层依赖、控制面/数据面分离、仓间解耦、legacy 约束、高风险区 | PR 触碰 src/ 时 |
| [`markdown-writing.md`](./markdown-writing.md) | Markdown 写作规范：文件命名、标题、图片、代码块、列表、表格、标点、目录结构、通顺/敏感词/术语 | PR diff 含 `.md` 文件变更时 |
| [`markdown-api-accuracy.md`](./markdown-api-accuracy.md) | 接口文档准确性：原型与头文件一致性、参数解释、代码样例 | PR 触碰 `docs/zh/api_ref/` 或文档涉及 `include/`、`pkg_inc/` 接口原型时 |
| [`pr-completeness.md`](./pr-completeness.md) | PR/Issue/实现三者吻合度、测试完备性、文档同步、模块目录文档同步（代码级目录文档补/更判断） | 所有 PR |
| [`gitcode-api.md`](./gitcode-api.md) | GitCode API 端点、认证、position 语义（检视工具链本身） | 提交检视意见时 |
