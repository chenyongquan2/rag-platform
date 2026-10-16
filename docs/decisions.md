# 决定日志

主人不在时按设计里的比较标准自己定下的事，每条写：定了什么、为什么、还考虑过什么。

## 待确认

（涉及花钱超预算、公开仓库、删除数据、改范围的事，暂停并写在这里。）

## 已定

### 2026-10-17 向量模型的名字写成 qwen/text-embedding-v4
网关上直接写 `text-embedding-v4` 返回 503（找不到可用通道）；`qwen/text-embedding-v4` 能调通。

### 2026-10-17 对话模型先用 deepseek-v4-flash
主人说先用最便宜的模型。网关的模型列表不带价格，deepseek-v4-flash 前期已经实测能用、按公开价格在列表里最便宜之一。网关会把它转到 deepseek-v4.1-flash，这是推理模型，会多花推理 token，估算花费时按输出 token 的上限算。

### 2026-10-17 P0 阶段应用跑在本机，Docker 只跑基础服务
ES、Redis、Collector、Jaeger 放进 Docker Compose；Python 程序用 `uv run` 在本机跑。原因：重排模型和 Hugging Face 缓存都在本机，调试快；WSL 内存保持默认 16 GB，少一个装 torch 的容器能省 2 到 3 GB。P8 再做应用的镜像。
