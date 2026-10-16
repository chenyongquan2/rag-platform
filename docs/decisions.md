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

### 2026-10-17 P0-01：依赖一起安装，没有版本冲突；torch 用 CPU 版
llama-index-core 0.14.25、langchain 1.4.5、langgraph 1.2.15、mcp 2.3.0、docling 2.140.0、sentence-transformers 6.1.0 放在同一个 `uv.lock` 里一次解析成功，没有改任何版本，docling 和 torch 也不用拆成可选依赖。torch 和 torchvision 从 PyTorch 官方的 CPU 源（download.pytorch.org/whl/cpu）安装，在 `[tool.uv.sources]` 里指定，所以要把它们写成直接依赖（uv 只对直接依赖按 sources 选源）。不这样做的话，Linux 上从 PyPI 装的 torch 会带几 GB 的 CUDA 库。还考虑过：直接用 PyPI 的 torch（Windows 上本来就是 CPU 版，但到了 Linux 容器和 CI 里体积大很多）。

### 2026-10-17 P0-01：锁文件只解析 Windows 和 Linux
`[tool.uv] environments` 只写了 win32 和 linux。开发在 Windows 上，容器和 CI 跑 Linux，没有 macOS 的需求；少解析一个平台，CPU 源里缺 macOS 包时也不会导致解析失败。以后要支持 macOS，在这里加一行再重新 `uv lock`。

### 2026-10-17 P0-01：多加两个小依赖 langgraph-checkpoint-sqlite 和 filelock
设计 16.7 要求 LangGraph 存档用 SQLite、摄取锁用文件锁，这两个包分别提供 SqliteSaver 和跨进程的文件锁。不自己写的原因：存档的表结构由 LangGraph 定，自己写容易和以后的版本不兼容；文件锁在 Windows 和 Linux 上的系统调用不一样，filelock 已经处理好了。

### 2026-10-17 P0-01：密钥扫描用 gitleaks，不用 detect-secrets
本机装了 Go 1.24，GOPROXY 是 goproxy.cn。pre-commit 用本机的 Go 从源码编译 gitleaks v8.30.1，第一次安装大约 7 分钟，之后直接用编译好的程序。试过一个假的 AWS 密钥，gitleaks 能报出来；对仓库全部历史提交扫了一遍，没有发现密钥。另外加了一个本地钩子：暂存区里出现 `.env` 或 `.env.*`（`.env.example` 除外）就不让提交。代价：别的机器要提交代码，也需要装 Go。

### 2026-10-17 P0-01：ruff 和单元测试的提交前检查用项目虚拟环境里的版本
钩子用 `uv run --frozen ruff ...` 和 `uv run --frozen pytest`，版本以 `uv.lock` 为准，不在 `.pre-commit-config.yaml` 里另写一份 ruff 版本，免得两处版本不一致。没有加去行尾空格、补文件末尾换行这类会自动改文件的钩子，因为它们会改到设计文档和 `docs/` 下主人维护的文件。

### 2026-10-17 P0-01：只有 kb/public/ 下的知识库默认对所有调用方开放
评审指出：`access.callers` 默认 `["*"]` 的话，私有知识库的配置只要漏写 `access`，所有调用方都能查到。现在的规则：
- 判断是不是公开知识库，只看配置文件相对 kb 根目录的路径是否以 `public/` 开头；`kb/private/public/x.yaml` 算私有。
- `kb/public/` 下的配置没写 `access` 时，`callers` 取 `["*"]`，所有已认证的调用方都能用。公开知识库仍然只写数据源就能用，符合设计 5.1。
- 其他目录（比如 `kb/private/`）下的配置没写 `access` 时，`load_kb_config` 直接报错，错误信息里写明要设置 `access.callers`。
- 在代码里直接构造 `KnowledgeBaseConfig` 而不给 `access` 时，`callers` 是空列表，谁都不能访问。

评审给了两个做法：私有配置漏写时报错，或者默认空列表、谁都不能访问。选报错，原因：默认空列表时知识库照常加载，只是所有调用都查不到内容，主人要排查一阵才知道是漏写了 `access`；报错在服务启动时就把文件名和要补的字段说清楚。

### 2026-10-17 P0-01：docstore 连接串指向一个目录，每个知识库一个文件
`DOCSTORE_URL=file://data/docstore`，`make_docstore(url, kb)` 读写 `data/docstore/<kb>.json`。一个连接串给所有知识库用；以后换成 Postgres 时，知识库名就是 LlamaIndex docstore 的 namespace。`file://` 和 `sqlite://` 后面的整段就是文件路径，相对路径从工作目录算起，不用 SQLAlchemy 那种三个斜杠、四个斜杠区分相对和绝对路径的写法，Windows 路径（`file://C:/...`）也能直接写。
