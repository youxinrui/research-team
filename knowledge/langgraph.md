# LangGraph 与断点续跑

LangGraph 用 StateGraph 描述有状态的 agent 工作流。节点是处理函数，边定义节点之间的流转，共享 State 在所有节点之间传递。条件边根据当前状态决定下一步走向，因此可以实现分派、循环和终止。

checkpoint 机制会在每个 super-step 完成后把状态保存到持久化后端。编译图时传入 checkpointer，调用时通过 config 提供 thread_id，同一个 thread_id 就对应同一个任务的状态历史。LangGraph 官方提供内存、SQLite 和 Postgres 等后端。

SqliteSaver 基于本地 SQLite 文件，零额外服务，适合单机 demo；当 agent 部署为多实例服务时，可以替换为 PostgresSaver，让不同进程共享状态。这体现了 checkpointer 作为可注入依赖的扩展性。

断点续跑的实现方式是：任务中断后，用相同的 thread_id 再次调用图，并传入 None 作为输入，图会从最近一次完成的节点继续执行。已经完成的节点不会重复运行。若中断发生在某个节点执行过程中，该节点会在恢复时重新执行，因此有副作用的工具需要保证幂等。
