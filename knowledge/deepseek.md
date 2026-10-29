# DeepSeek 模型与 OpenAI 兼容接口

DeepSeek 提供多个对话模型，可通过 OpenAI 兼容的接口调用。客户端把 base_url 指向服务地址，把模型名填入 model 字段，就可以复用 OpenAI SDK、LangChain 的 ChatOpenAI 等现成封装，而不必编写专用的请求代码。

DeepSeek 的接口主要面向文本生成和工具调用，不提供 embedding 接口。因此做 RAG 时需要单独接入向量化服务。国内常用的选择包括硅基流动的 BAAI/bge-m3、阿里百炼的 text-embedding-v3，以及智谱的 embedding-3，它们都提供 OpenAI 兼容的 /v1/embeddings 端点。

接入 embedding 服务时，需要在索引构建和查询时使用同一个模型与同一维度。如果更换了 embedding 模型，必须重新构建索引，否则查询向量和文档向量的维度或语义空间不一致，会导致检索结果错误。

调用 OpenAI 兼容接口时还应配置合理的超时和重试次数。网络不稳定时，重试可以吸收瞬时失败；但重试次数过多会显著拉长任务时间，因此需要与外层的最大步骤限制配合使用。
