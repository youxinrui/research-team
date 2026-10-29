# MCP 模型上下文协议

MCP（Model Context Protocol）是连接模型应用与外部能力的开放协议。它把能力提供方抽象为 MCP Server，把使用方抽象为 MCP Client，Host 负责承载模型和客户端。通信基于 JSON-RPC，常见传输方式包括本地 stdio 和远程 SSE 或 HTTP。

MCP Server 可以暴露三类能力。Tools 是模型可以调用的函数，例如搜索、数据库查询、文件操作；Resources 是可供读取的上下文数据，例如文档、配置、日志；Prompts 是预定义的提示模板，供用户或应用选择。工具需要通过名称、描述和输入 schema 声明自己，客户端据此决定何时调用。

MCP 与 function calling 的关系是：MCP 定义了工具从哪里来、如何被发现和调用，function calling 定义了模型如何表达调用意图。两者可以组合使用，MCP Server 提供的工具可以被转换为模型可调用的工具列表。

接入 MCP 时要注意安全边界。工具应有白名单和最小权限，危险操作需要人工确认；远程调用要设置超时和重试；所有工具调用应记录审计日志。把搜索能力封装成 MCP Server，可以让多个 agent 或应用复用同一套工具，而不必各自实现。
