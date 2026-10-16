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

### 2026-10-17 P0-03：HF 文档集下载 Hub 自动转换出来的 parquet，不下载原始 CSV
`m-ric/huggingface_doc` 的 main 分支里只有一个 `huggingface_doc.csv`（21,954,601 字节）。Hugging Face Hub 会自动把数据集转成 parquet，放在 `refs/convert/parquet` 分支（`default/train/0000.parquet`，10,058,873 字节）。选 parquet 的原因：
- 任务要求下载 parquet。
- parquet 带列类型，读的时候不用处理正文里大量的换行和引号转义；文件也小一半。
- 清单里写明下载哪个提交（`ec31afec…`），以后 Hub 重新转换出新的提交，也不影响我们下载的这一版。清单的 `note` 里记了它是从 main 的哪个提交（`1b83935…`）转出来的。

还考虑过：直接下载 CSV。好处是用户上传的原文件，不依赖 Hub 的转换。如果转换分支的这个提交以后被删掉，就改成下载 CSV，需要加一段读 CSV 的代码。评测题集 `m-ric/huggingface_doc_qa_eval` 的 main 分支本身就是 parquet，直接用 main 的提交 `5f70aa9…`。

### 2026-10-17 P0-03：自己用标准库下载，不用 huggingface_hub 的 hf_hub_download
下载地址按 Hub 的格式自己拼：`<endpoint>/datasets/<repo>/resolve/<提交 id>/<文件路径>`，用 Python 标准库 urllib 下载。文件存到 `<HF_DATA_DIR>/<repo>/<提交 id>/<文件路径>`。原因：
- huggingface_hub 的缓存目录用 blobs 加 snapshots 符号链接。Windows 上没有建符号链接的权限时，它会改成复制文件，目录结构和 Linux 上不一样。
- 它不会按我们的清单核对大小、sha256 和行数，这一步反正要自己写。
- 自己拼地址，单元测试可以把地址换成 `file://` 的本地目录，不联网就能测下载、缓存和核对。

还考虑过：用 huggingface_hub。它已经是间接依赖，也认 `HF_ENDPOINT` 环境变量。服务配置里新加的 `hf_endpoint` 字段读的也是 `HF_ENDPOINT`，两边设一次就都走镜像。

### 2026-10-17 P0-03：新加依赖 pyarrow
锁文件里原来没有 pyarrow。读 parquet 必须用它，pandas 读 parquet 也要靠它。版本是 26.0.0。P0-02 也在改 `uv.lock`，两个分支先后合并到 main 时，`uv.lock` 可能冲突，到时重新跑一次 `uv lock` 就能解决。

### 2026-10-17 P0-03：doc_id 写成「知识库名:相对路径」，HF 文档的相对路径就是 source 列
设计 5.4 要求 doc_id 由知识库、相对路径和单元组成。具体格式：`hf_docs:huggingface/transformers/blob/main/docs/source/en/bert.md`。文件里还分单元时（比如 CHM 里的页面），后面加 `#<单元>`。用冒号隔开知识库名，因为路径里本身有斜杠。路径里的反斜杠统一换成斜杠，绝对路径和带 `..` 的路径直接报错，这样同一个文件在 Windows 和 Linux 上得到同一个 id。

数据集的 `source` 列就是这篇文档在 GitHub 上的路径，所以 `source_url` 写成 `https://github.com/` 加上这个路径，前缀写在清单的 `doc_url_prefix` 里。实查 2,647 个 `source` 没有空值，也没有重复。以后出现重复时直接报错，不自动加序号：加序号的话，id 会随行的先后顺序变化，增量更新就认不出是同一篇。评测题的 `source_doc` 用同一个函数算 doc_id，65 题的来源文档都在文档集里。

### 2026-10-17 P0-03：limit 取文件里的前 N 行，不先按路径排序
知识库配置里的 `limit: 50` 取的是清单指定的那个 parquet 文件里的前 50 行。文件按 sha256 核对过，所以每次取到的 50 篇都一样。没有先按路径排序再取，因为按字母顺序排，前 50 篇全是 `gradio-app` 仓库的文档（共 469 篇，排在 `huggingface` 前面），而文件原来的顺序是各个仓库混在一起的。

### 2026-10-17 P0-03：清单里的大小、sha256 和行数由第一次下载记下，再提交
`data/manifests/hf_docs.json` 里的仓库、提交 id、许可、来源网址和预期行数（2,647 和 65）是手写的，大小、sha256 和每个文件的行数一开始留空。运行一次 `python -m rag.data.hf download --pin`，程序从下载下来的文件算出这三项写回清单。之后的下载和读取都要求这三项齐全，而且和文件一致。第一次记下的 sha256 和 Hub 接口返回的 LFS 对象 id（也是 sha256）比过，两个文件都一样。

下载先写到临时文件，核对通过才改名放进缓存，所以缓存里不会留下核对没通过的文件；两个 checkout 共用一个缓存目录、同时下载也不会读到写了一半的文件。`HF_DATA_DIR` 默认是 `data/raw/hf`（相对当前目录，在 .gitignore 里）。这次在 P0-03 的 worktree 里运行时，把它设成了 `C:\workspace\rag-platform\data\raw\hf`，主 checkout 以后用默认值就能直接用这份下载。

### 2026-10-17 P0-03 评审后：doc_id 的路径里的 # 和 % 改写成 %23 和 %25
doc_id 用 `#` 隔开文件路径和单元（比如 CHM 里的页面名）。如果文件名里本身有 `#`（比如 `c#.md`），读 doc_id 的程序就分不清哪个 `#` 是分隔符。现在 `make_doc_id` 把路径里的 `#` 写成 `%23`，`%` 写成 `%25`，单元里的 `#` 不改。这样路径之后的第一个 `#` 一定是分隔符，两个不同的路径也不会算出同一个 id。路径里的 `a//b`、`a/./b` 先合并成 `a/b`。

还考虑过：路径里有 `#` 就报错。缺点是以后接本地目录、PDF 时，文件名里有 `#` 的文档整份读不进来。HF 文档集的 2,647 个路径里没有 `#`、`%`、`//` 和 `/./`，改完以后它们的 doc_id 和改之前一样（实查过）。

### 2026-10-17 P0-03 评审后：提交在仓库里的清单，相对路径从仓库根目录算
知识库配置里写的 `manifest: data/manifests/hf_docs.json`、`load_eval_questions` 收到的相对路径、命令行 `--manifest` 的默认值，都从仓库根目录算（`rag.config.repo_path`），不从当前目录算。原因：清单是提交在仓库里的文件，在别的目录下运行程序也应该找到同一份。命令行里明确传的 `--manifest` 还是按当前目录算，和一般命令行工具一样。

P0-01 定的规则（连接串里的相对路径从当前目录算）不变：`data/docstore`、`HF_DATA_DIR` 这些是运行时写出的数据，不是仓库里的文件。

### 2026-10-17 P0-02：镜像版本和从哪里拉
- Elasticsearch 用 9.5.5，是当天能拉到的最新 9.x 版本，从 Elastic 自己的镜像仓库 docker.elastic.co 直接拉成功，没用镜像源。
- Redis 8.10.2-alpine、OTel Collector contrib 0.162.0、Jaeger 2.22.0 在 Docker Hub 上。Docker Desktop 里已经配了国内镜像源（docker.m.daocloud.io、mirror.baidubce.com 等），直接 `docker pull` 就拉成功了，compose 里写的仍是 Docker Hub 上的原名。
- Jaeger 用 v2 的 `jaegertracing/jaeger` 镜像，不用 v1 的 `jaegertracing/all-in-one`：v1 的最后一个版本是 2025-12 的 1.76.0，已经不再维护。v2 不带配置文件启动时就是一体化模式，trace 存在内存里。v2.22 只提供 `/api/v3/...` 查询接口，v1 的 `/api/traces` 返回 404。

### 2026-10-17 P0-02：本机端口都用「标准端口加 20000」，只绑 127.0.0.1
- 这台机器上 6379、4317、4318、13133、16686 已经被别的项目的容器（还有一个本机的 Redis 进程）占着。停掉别人的容器会影响别的项目，所以不用这些端口。
- 这台机器的 Windows 动态端口范围是 1024 到 15000（Windows 默认从 49152 开始），Hyper-V 会在这个范围里成段保留端口；当时 6380 到 6479 被保留，compose 绑 6380 直接失败。保留哪几段在重启后会变，所以端口全部放在 15000 以上：ES 29200、Redis 26379、OTLP gRPC 24317、OTLP HTTP 24318、Collector 健康检查 33133、Jaeger 页面 36686。
- `ServiceSettings` 的默认连接串跟着改，而且写 127.0.0.1 不写 localhost。实测 Python 连 localhost 每次多等约 2 秒（先试 IPv6 的 ::1，被拒绝后才试 127.0.0.1），连 127.0.0.1 约 10 毫秒。P0-01 的默认值测试 `test_service_settings_defaults` 因此改了。
- 还考虑过：在 compose 里用环境变量配端口。换一台机器端口冲突时更方便，但现在只有这一台开发机，先不加。

### 2026-10-17 P0-02：本机 ES 关闭安全功能和机器学习
- `xpack.security.enabled=false`：本机开发不用 TLS 和密码。端口只绑 127.0.0.1，别的机器连不上。部署到服务器时（P8）必须打开，连接串写成 `https://用户名:密码@主机`，`make_es_client` 会把它转成 basic auth。
- `xpack.ml.enabled=false`：项目不用 ES 的机器学习功能，关掉少占内存、启动更快。
- 许可证写明 `xpack.license.self_generated.type=basic`。

### 2026-10-17 P0-02：索引映射按 ES 9.5.5 实测确认，另外多写了三项
- 9.5.5 接受设计 7.3 的写法：`int8_hnsw` 加 `rescore_vector.oversample=3`，取回的映射和写进去的一样。
- HNSW 的 `m=16`、`ef_construction=100` 是 ES 现在的默认值，也写进映射，以后升级时默认值变了，已有配置不跟着变。
- 9.5 新建的索引默认 `index.mapping.exclude_source_vectors=true`，查询结果的 `_source` 里不带向量。映射里显式写成 true：检索用不到向量原文，返回的数据少很多。以后要从 ES 取回向量时，在查询里单独要。
- `dynamic: strict`：写入映射里没有的字段时 ES 直接报错，不自己猜类型。比如把 `acl_groups` 拼错，写入时就会失败；如果让 ES 自动加字段，权限过滤会一直匹配不到，却没有任何报错。P0-06 写入时要多存字段，先在 `rag.index.mapping` 里加上。还考虑过 `dynamic: false`（映射里没有的字段只存进 `_source`、不建索引）：写入不报错，拼错的字段名要等查询结果不对才发现。
- 向量类型做成 `VectorIndexOptions`，可以选 int8_hnsw、bbq_hnsw、flat 等，留给 P2 做消融；bbq_disk 要 Enterprise 许可证，不提供。

### 2026-10-17 P0-02：数据代号存在会淘汰键的 Redis 里，键丢了就从当前毫秒时间重新开始
- 问题：redis-cache 按计划配置成 allkeys-lru，也不存盘。数据代号这个键可能被淘汰，Redis 重启后也会丢。如果丢了以后从 0 或 1 重新数，就会回到以前用过的值，用这个旧值做键的缓存会再次被命中，返回已经改过或删掉的内容。
- 做法：键不存在时先写入 Redis 服务器当前 Unix 时间的毫秒数，再读或加 1。判断键在不在、读服务器时间（`TIME`）、写入、加 1 放在同一个 Lua 脚本里，Redis 执行脚本时不会插进别的命令。这样新值一定比以前所有的值大，前提有两个：一个知识库加 1 的次数少于这段时间经过的毫秒数（每次摄取至少几秒，实际一定满足）；Redis 服务器的时钟没有往回调。读的时候键不存在也会写入，保证连续几次读到同一个值。
- 评审后改成用 Redis 服务器的时钟，不用调用方本机的时钟：几个进程可能跑在时钟不一致的机器上，某台机器的时钟慢了，它重新写入的值就可能比以前用过的值小。所有进程连的是同一个 Redis，用它的时钟就只有一个时钟。
- 代价：数据代号是 13 位的大数，不是 1、2、3 这样的小数。
- 还考虑过：改成 volatile-lru，只淘汰设了过期时间的键，数据代号不设过期时间就不会被淘汰；但这要求所有缓存键都设过期时间，而且和计划里写的 allkeys-lru 不一致。另起一个不淘汰键的 Redis 专门存数据代号，要多一个容器。存在 ES 索引的 `_meta` 里，每次查缓存都要先多查一次 ES。

### 2026-10-17 P0-02：切换别名和删除旧索引怎么防出错
- `switch_alias` 在一个 `_aliases` 请求里同时做「从旧索引去掉别名」和「给新索引加上别名」，ES 对这个请求要么全部执行、要么全部不执行。集成测试里两个线程不停地按别名查询，主线程同时来回切换 20 次，每次查询都只看到一个索引，没有出现查不到或者两个索引都查到的情况。
- 调用方可以传 `expected_current`，表示「我确认过别名现在指向 v1」。去掉 v1 上的别名时带 `must_exist`，如果别名已经被别人切走，ES 拒绝整个请求，别名保持原样。
- 评审发现的问题：调用方以为别名已经在目标索引上（`expected_current` 等于要切到的索引）时，原来的代码省掉了带 `must_exist` 的那一步，请求只剩「加别名」。如果别名其实已经被切到别的索引，加完以后别名同时指向两个索引，查询会查到两份结果。现在无论是否等于目标，都先带 `must_exist` 去掉预期索引上的别名，再加到目标上；ES 允许同一个请求里对同一个索引先去掉再加回。
- ES 拒绝请求时按错误类型区分：别名不在预期索引上（`aliases_not_found_exception`）、预期索引已经不存在，都报「别名被别人改过」；要切到的索引在检查之后被删了，报索引状态错误；其他错误原样抛出，不当成别名冲突。
- 两个进程同时切换同一个知识库的别名，要靠摄取锁防止（P0-06），这个函数自己不加锁。
- `delete_kb_index` 只接受本知识库带版本号的完整索引名，别名、通配符、别的知识库的索引都不删；索引上还挂着任何别名时也不删，要先把别名切走。检查和删除是两次请求，中间被切换的情况同样靠摄取锁防止。
- 命令行的 `switch` 先切别名，再把数据代号加 1。顺序反过来的话，加 1 之后、切换之前写进的缓存会带着新代号，内容却是旧索引的。如果切完别名时 Redis 连不上，命令报错并提示重跑同一条 `switch` 命令：重跑时别名已经指向目标，不再变动，只把数据代号加 1。

### 2026-10-17 P0-02：实测的内存和启动时间
- `docker stats`（索引里还没有数据）：ES 约 2.5 GB，Jaeger 约 15 MB，Collector 约 37 MB，Redis 约 8 MB。设计 16.5 估的是 ES 约 3 GB、Collector 和 Jaeger 合计 0.3 到 0.6 GB；Jaeger 在内存里存的 trace 多了以后会变大。没有给容器设内存上限。
- 所有容器从停止状态启动到健康检查全部通过，用了 40 秒到 2 分 20 秒，`scripts/dev_up.sh` 默认最多等 300 秒。
- Collector 镜像里没有 shell 和 HTTP 客户端，写不了容器健康检查，`dev_up.sh` 从本机访问它的健康检查端口（33133）。

### 2026-10-17 P0-05：向量客户端自己写，继承 LlamaIndex 的 BaseEmbedding，用 httpx 调网关
`rag.embed.GatewayEmbedding` 继承 LlamaIndex 的 `BaseEmbedding`，用 httpx 向网关的 `/embeddings` 发请求。摄取和查询都调用 `make_embedding` 创建它：模型名、维度、文本前缀、每批条数读同一份配置，只有超时、重试次数和重试间隔按 `ingest`、`query` 两种用途分开设。它是 `BaseEmbedding` 的子类，单元测试里放进 `IngestionPipeline`，同步和异步都能用。

按五条标准比了三种做法：

| 做法 | 能拿到 usage | 重试能按配置设、只有一层 | 不加新依赖 | 不联网就能测 | 能用 `qwen/text-embedding-v4` 这种模型名 |
|---|---|---|---|---|---|
| LlamaIndex 的 OpenAILikeEmbedding | 不能 | 不能，有两层 | 不满足，要加两个包 | 能 | 能 |
| 自己写，内部调 openai SDK | 能 | 次数能设，等待时间是 SDK 里的固定值 | 满足 | 能 | 能 |
| 自己写，用 httpx（选这个） | 能 | 能 | 满足 | 能 | 能 |

- LlamaIndex 那一行是按以前读过的源码说的，这次没有装包核对（我的推断）：`llama-index-embeddings-openai` 的 `get_embeddings` 只返回向量，把响应里的 usage 扔了，账本就记不到真实 token 数；openai SDK 自己重试一层，LlamaIndex 外面再用 tenacity 重试一层，两层的次数相乘；它还会把文本里的换行换成空格，这是一个不显式写在我们配置里的默认行为。`OpenAIEmbedding` 只认 OpenAI 自己的模型名，所以要用 `llama-index-embeddings-openai-like` 里的 `OpenAILikeEmbedding`，两个包的版本要和 llama-index-core 一起升级。
- openai SDK 已经随 langchain-openai 装好了，但它的重试等待时间从 0.5 秒起、最长 8 秒，配置改不了。
- httpx 本来就装了（0.28.1），这次在 `pyproject.toml` 里写成直接依赖，`uv.lock` 只多了两行。

代价：自己维护约 300 行代码，OpenAI 接口以后加了字段要自己改。以后要用的接口不是 OpenAI 格式，或者要用 LlamaIndex 的多协程并发（`num_workers`），再重新比较。

### 2026-10-17 P0-05：网关一次最多收 10 条文本，默认每批 10 条；超时、重试和等待上限按用途分开
探测方法：不重试，向网关依次发 1 条、10 条、11 条很短的文本（`t0`、`t1`……）。10 条成功；11 条返回 HTTP 400，错误信息是 batch size 不能大于 10。所以 `embed_batch_size` 默认 10。配置允许的上限 2048 是 LlamaIndex 自己的上限，设成大于 10 时网关返回 400，第一次请求就报错，不会重试。

同时量到：单条文本一次请求约 0.6 秒，第一次 1.75 秒（包括建立 TLS 连接）；10 条一次请求 2.6 秒（也是新连接）。同一段文本连发三次，三个向量逐位相同。

超时和重试（评审后改过，见表后）：

| 用途 | 单次超时 | 最多重试 | 两次之间等多久 | 最多等多久 | 总时限 | 网关一直不通时最坏等多久 |
|---|---|---|---|---|---|---|
| 摄取 `ingest` | 60 秒 | 4 次 | 2、4、8、16 秒 | 60 秒 | 不限 | 每批约 330 秒 |
| 查询 `query` | 3 秒 | 1 次（配置不允许大于 1） | 0.2 秒 | 1 秒 | 7 秒 | 约 6.2 秒 |

会重试的错误：超时、连接失败、HTTP 408、429 和 5xx。网关带了 `Retry-After`（秒数）时按它等；它要求等的时间超过这个用途的「最多等多久」，程序不等，直接抛 `EmbeddingError`。查询还有总时限：每次请求的超时取「单次超时」和「离总时限还剩的时间」中小的那个；如果等完再试会超过总时限，也直接报错，不再等。其他错误第一次就报错：400、401、403、404 等 4xx，回答不是 JSON，向量条数不对，维度不对。直接写 `text-embedding-v4` 时网关一直返回 503，会按 5xx 重试到次数用完再报错，摄取时要多等约半分钟才看到错误；没有为它单独处理。

评审后改的：第一版两种用途共用一个 30 秒的等待上限，查询碰到一次带 `Retry-After: 30` 的 429，就要等 30 秒，和设计 16.2「查询用短超时」矛盾。现在每种用途有自己的上限，查询另有 7 秒的总时限。`Retry-After` 超过上限时直接报错，不改成少等几秒再试：网关要求等 30 秒，等 1 秒再试多半还是 429，只是多发一次请求。

### 2026-10-17 P0-05：L0 缓存的键里加上维度和文本前缀，向量按 float32 存
设计 12.1 写的键是「向量模型名 + 文本哈希」。实际的键分两部分：模型部分是 `qwen/text-embedding-v4@1024`，配了文本前缀时再加前缀的 sha256 前 12 位；文本部分是原文的 sha256。原因：同一个模型名，换了维度（text-embedding-v4 支持 64 到 2048 维）或换了前缀，同一段文本的向量就不一样了。只用模型名做键，改了维度以后，缓存会返回旧维度的向量。

- 向量按 float32 存，1024 维占 4 KB。ES 的 dense_vector 本来就按 float32 存。网关新返回的向量也先转成 float32 再交给调用方，所以同一段文本不管是不是从缓存读的，拿到的数字都一样。
- 同一批里重复的文本只发一次。查询和摄取用同样的键：两边前缀相同，同一段文本的向量相同（llm 测试实测逐位相同）。
- 摄取用 SQLite 文件 `data/cache/embeddings.sqlite3`（`data/cache/` 已在 .gitignore 里），WAL 模式。每次请求拿到的向量用一个事务写进文件，再发下一次请求，所以摄取中途停下，重跑时只算还没写进缓存的文本。

还考虑过：用 LlamaIndex `BaseEmbedding` 自带的 `embeddings_cache`。它的键是原文本身，不带模型名；一条一条读写，一批 10 条要读写 10 次；存的值是「随机 uuid 对应向量」，以后的版本可能改格式。所以自己写了 `CachedEmbedding`，包在客户端外面，一批只读一次、写一次。

### 2026-10-17 P0-05：查询缓存默认用 redis_url，Redis 出错时当作没命中
`embed_query_cache_url` 留空时用 `redis_url`，因为设计 16.7 规定缓存只用一个 Redis；单元测试或者本机没开 Redis 时可以设成 `memory://`。键是 `rag:l0:<模型部分>:<sha256>`，30 天过期。compose 里的 Redis 设了 allkeys-lru，内存满了会删掉最久没用的键。

Redis 连不上或读写报错时，程序记一条警告，当作没命中，查询照样去网关算向量；连接和读写的超时都是 0.5 秒。原因：缓存只用来省钱、省时间，Redis 出故障不该让查询失败。

没有用 P0-02 写的 `make_redis_client`：它设了 `decode_responses=True`，Redis 返回的值会被当成文本解码，二进制的向量会解码失败。所以缓存自己用 `Redis.from_url` 建客户端，不解码。

### 2026-10-17 P0-05：账本记哪些请求；发请求之前先预留预算，同时最多 4 个请求
网关每返回一次 2xx，账本就加一行。token 数从响应的 usage 读；没有 usage，或者回答解析失败时，按字符数估算，这一行写 `usage_estimated: true`。请求发出去以后读超时或者连接断开（httpx 的 `ReadTimeout`、`ReadError`），网关可能已经算完并扣了费，也按字符数估算记一行，并写上 `error` 字段（评审后加的）。连不上、4xx 和 5xx 不记，因为网关没有处理这个请求（我的推断，没查到网关的计费说明）。

并发和预算（评审后改的）：评审实测 `IngestionPipeline.arun` 把所有批次同时发出去，一次有 20 个请求在等网关；第一版每个请求各自查一次预算，大家都看到「还没超」，就一起发出去，合起来可能超过 28 元。现在：

- 每个客户端同时最多 `embed_max_concurrency`（默认 4）个请求在等网关，其余的在线程里排队。
- 每个请求发出之前，先用 `CostLedger.reserve` 预留这次请求的估算金额（把字符数当 token 数，比实际多）。查和占在同一把锁里做：读账本里已经花的钱，加上正在进行的请求预留的金额，再加这次的估算，超过 `budget_stop_cny`（28 元）就抛 `BudgetExceeded`，不发请求；不超过就记下预留的金额。请求结束以后删掉这笔预留。同一进程里写同一个账本文件的 `CostLedger` 对象共用这份记录，`check_budget` 也把它算进去。
- 不同进程之间看不到对方预留的金额。几个进程同时调真实模型时，最多多花各自正在进行的那几个请求的钱：一个请求 10 块、每块 2,000 字符时估算约 0.01 元，每个进程最多 4 个，离 30 元的总预算还差 2 元，够用。

每次查预算都读整个账本：HF 文档全量摄取约 500 万 token，按每块约 400 token、每次 10 块估算，约 1,250 次请求，账本也只有一千多行；用 1,250 行的账本实测，查一次预算约 13 毫秒，比一次请求的两秒多小得多。以后行数多到拖慢请求，再改成把总数存在内存里。

每行的字段：time、kind、model、prompt_tokens、completion_tokens、total_tokens、cny，再加调用方自己的字段（profile、inputs、usage_estimated、error）。写的时候加文件锁，几个进程同时写，两行也不会混在一起。最后一行如果因为进程崩溃只写了一半，读的时候跳过并记一条警告。

### 2026-10-17 P0-05 评审后：账本不进 git，所有 checkout 和 worktree 共用仓库外的同一个文件
第一版把 `data/cost/ledger.jsonl` 提交进 git，用 `merge=union` 合并。评审指出：几个 worktree 同时调真实模型时，每个 worktree 的账本只有自己的花费，28 元的停止线看不到别的 worktree 花的钱。现在：

- `cost_ledger_path` 默认是「主 checkout 的上一级目录」下的 `rag-platform-data/cost/ledger.jsonl`，在这台机器上就是 `C:/workspace/rag-platform-data/cost/ledger.jsonl`。程序读 worktree 里的 `.git` 文件（内容是 `gitdir: <主 checkout>/.git/worktrees/<名字>`）找到主 checkout，所以 worktree 不管放在哪个目录，算出来的都是同一个文件。没有 git 信息时（比如装好的包），用代码所在仓库目录的上一级。目录不存在时，写第一行时自动创建。`COST_LEDGER_PATH` 仍然可以改。
- `data/cost/` 放进 .gitignore，删掉 `.gitattributes`；第一版提交的 6 行已经挪进共用的文件。
- 账本不在 git 里，花了多少钱由协调的人写进 `docs/progress.md`。

还考虑过：每个 checkout 一个账本，靠 git 合并加起来。缺点就是上面说的，合并之前，预算检查看不到别的分支花的钱。

### 2026-10-17 P0-05：对话模型的价格先写一个偏高的估计
价格表 `model_prices` 里，`deepseek-v4-flash` 先写成输入 2 元、输出 8 元每百万 token。网关的模型列表不带价格，这次没有查到真实价格。故意写高：预算检查只会提前停下，不会花超了才停。P1-01 第一次真实调用以后，按网关的实际扣费改价格。向量模型按设计 20.2 写 0.5 元每百万 token。价格表里没有的模型，创建客户端时直接报错，一次请求都不发。

### 2026-10-17 P0-05：网关不可用或预算用完时，手动改配置用假向量，程序不自动切换
设 `EMBED_BACKEND=fake` 以后，摄取和查询都用 `FakeEmbedding`（维度等于 `embed_dim`）：不发请求，不写账本，也不读写 L0 缓存。没做「网关出错就自动换成假向量」：假向量和真向量之间算相似度没有意义，自动切换以后，同一个索引里会混进两种向量，查出来的结果是错的，程序却不报错。用假向量建的索引只能用假向量查，P0-06 建索引时要记下用的是哪种向量。

### 2026-10-17 P1-01：对话模型的花费在 httpx 传输层记进账本，不用 LangChain 回调
`make_chat_model` 用 `init_chat_model` 建 `ChatOpenAI`，同时传进去两个自己建的 httpx 客户端（同步、异步各一个），客户端的传输层是 `LedgerTransport` 和 `AsyncLedgerTransport`。传输层是 httpx 客户端里真正把请求发到网络上的那个对象，客户端每发一次请求都调用它一次。openai SDK 每发一次 HTTP 请求都经过这一层，包括 SDK 自己的重试：

- 发出之前，按请求估算金额，用 P0-05 的 `CostLedger.reserve` 预留。已花的、正在进行的请求预留的、这次估算的三项加起来超过 `budget_stop_cny`（28 元），就抛 `BudgetExceeded`，请求不发出去。估算是上限：请求体的字符数当输入 token 数，请求里的 `max_completion_tokens` 当输出 token 数。所以 `max_tokens` 越大，每次预留越多：online 配置的 4096 个 token 按估计价约 0.033 元。
- 网关返回 2xx 以后，从响应的 usage 读 token 数，账本加一行；流式请求读最后一个事件里的 usage。读超时、连接中途断开、流式回答没读完就被关掉（用户关了页面），网关可能已经生成并扣了费，按估算记一行，带 `error` 字段。连不上、4xx 和 5xx 不记，和 P0-05 一样。
- 每行除了 P0-05 的字段，还有 `profile`、`reasoning_tokens`、`stream`、`served_model`（网关实际用的模型，现在是 deepseek-v4.1-flash）、`price_estimated`。
- 网关的错误回答里如果原样带出了 API key，传输层先把它换成 `***`，再交给 SDK 拼进异常信息。

按五条标准比了三种做法：

| 做法 | SDK 自己重试的每一次都预留、都在账本里写一行 | invoke、stream、结构化输出、同步异步都覆盖 | 能读到推理 token | 代码量 | 换成非 OpenAI 格式的接口 |
|---|---|---|---|---|---|
| LangChain 回调：开始时预留，结束时在账本里写一行 | 不能 | 能，但回调里抛的异常默认只写日志、请求照发，要设 `raise_error=True` | 能 | 约 100 行 | 能用 |
| 继承 `ChatOpenAI`，改写 `_generate`、`_stream` 等四个方法 | 不能 | 能 | 能 | 约 150 行 | 不能用，也不能再用 `init_chat_model` |
| httpx 传输层（选这个） | 能 | 能，不依赖 LangChain 内部怎么调 | 能，直接读网关返回的 usage | 约 300 行，要自己解析流式回答的 SSE 事件 | 要改这一层 |

- SSE（Server-Sent Events，服务器推送事件）：服务器在一个 HTTP 连接里一段一段地往客户端发数据，每段是一行 `data: {...}`；OpenAI 格式的流式回答就是这样发的，最后一段带 usage。
- 回调和继承这两种做法看不到 SDK 内部的重试：第一次请求读超时、SDK 重发，回调只收到最后一次的结果。第一次如果已经被网关计费，账本里就少一笔。P0-05 定过「读超时也记一行」，只有传输层能做到。
- 代价：这一层依赖 OpenAI 的响应格式（usage 字段、SSE 的 `data:` 行）。换成别的格式的接口时，要改这一层，或者改用回调。
- 外层客户端换了传输层以后，httpx 不再读环境变量里的代理设置。这台机器设了 `HTTPS_PROXY`，所以传输层里面再用一个普通的 `httpx.Client` 发请求，代理和 `NO_PROXY` 照样生效，和向量客户端一致。每次请求的超时仍是 SDK 设的值。
- 网关压缩过的回答，传输层先解压再交给 SDK，否则读不到 usage。

### 2026-10-17 P1-01：结构化输出默认 function_calling，实测网关不支持 json_schema
实测方法：cheap 配置，同一个问题（「7 是质数」这句话对不对），Pydantic 类 `Verdict` 有 `is_true` 和 `reason` 两个字段，两种方式各调一次。

| 方式 | 结果 | 耗时 |
|---|---|---|
| `function_calling` | 返回正确的 `Verdict` | 1.5 秒 |
| `json_schema` | 网关返回 HTTP 400：`This response_format type is unavailable now`；账本里没有写这一次 | 0.3 秒 |

所以 `chat_structured_output_method` 默认 `function_calling`，和设计 11.1 一致。`ChatOpenAI` 用 `function_calling` 时，请求里只放这一个工具，并用 `tool_choice` 指定模型必须调用它；deepseek-v4.1-flash 是推理模型，实测也接受指定的 `tool_choice`。这次调用输入 325 个 token（工具的 JSON Schema 占了大部分），输出 76 个，其中推理 11 个。配置里还允许 `json_schema` 和 `json_mode`，换成支持它们的网关或模型时可以改；`json_mode` 这次没有测。

`structured_output(model, schema)` 包了一层：模型没有调用工具时，LangChain 返回 None；参数不符合 Pydantic 类时，LangChain 把错误放在结果里。这两种情况都改成抛 `StructuredOutputError`（类别 `structured_output_failed`）。不让 None 往下传：路由节点拿到 None，不知道该走哪条路，错误会在后面的节点才暴露。`FakeChatModel` 只模拟 `function_calling`。

### 2026-10-17 P1-01：三套配置的超时、重试和输出上限；推理 token 的实测
| 配置 | 用在哪 | 单次超时 | 最多重试 | `max_tokens` | 最多按 Retry-After 等多久 | 网关一直不回应时最坏等多久 |
|---|---|---|---|---|---|---|
| `online` | 写回答 | 30 秒 | 1 次 | 4096 | 2 秒 | 约 61 秒 |
| `cheap` | 改写、路由、判断够不够 | 20 秒 | 1 次 | 2048 | 2 秒 | 约 41 秒 |
| `offline` | 出题、裁判 | 180 秒 | 4 次 | 8192 | 60 秒 | 约 15 分钟 |

三套配置默认都用 `chat_model`（deepseek-v4-flash），`temperature` 都是 0。每套是环境变量里的一个 JSON，只写要改的键，其余用这套配置的默认值，例如 `CHAT_ONLINE='{"timeout_s": 10}'`。

2026-10-17 实测四次调用（`tests/llm/test_chat_gateway.py`，每次输出上限 1024 个 token）：

- online 配置问「法国首都是哪」，要求只答一个词：1.5 秒返回 Paris。输入 42 个 token，输出 19 个，其中推理 17 个。
- 流式让模型从 1 数到 10：收到 15 个分块，4 个带正文，其余是推理内容、结束标记和 usage。第一个带正文的分块 0.8 秒到，全部收完 0.84 秒。账本里这一行的 usage 来自最后一个事件，不是估算。以后问答页面用 SSE 边生成边显示时，可以直接用 `.stream()`。
- 推理 token 已经算在 `completion_tokens` 里：`total_tokens` 等于输入加 `completion_tokens`，网关另在 `completion_tokens_details.reasoning_tokens` 里给出推理 token 数，LangChain 放进 `usage_metadata.output_token_details.reasoning`。账本按 `completion_tokens` 算钱，另记 `reasoning_tokens`。如果别的网关的 `total_tokens` 显示推理 token 没算进 `completion_tokens`，程序把它加上再算钱。网关返回的推理文字（`reasoning_content`）`ChatOpenAI` 不保留，我们只需要 token 数，不需要推理文字。
- cheap 配置把 `max_tokens` 改成 16，让模型列出前 20 个质数：16 个 token 全部用在推理上，正文是空的，`finish_reason` 是 `length`。这说明两件事：网关遵守 `ChatOpenAI` 发的 `max_completion_tokens`，所以预留时把它当输出上限是对的；`max_tokens` 必须给推理留够，所以三套配置最少 2048。

为什么这样取值：

- 不流式调用时，网关生成完才返回，单次超时就是整个生成最多花多久。实测短回答 1.5 秒。online 定 30 秒：按每秒约 50 个 token 算（我的推断，这次没有测长回答的速度），30 秒能生成约 1500 个 token（推理加正文）。cheap 只输出一个小 JSON，定 20 秒。offline 出题和裁判的提示词长，任务不着急，定 180 秒、重试 4 次。
- 设计 11.1 写的是在线回答「几秒超时」，现在的 30 秒和最坏 61 秒都比这长。原因是推理模型生成一个完整回答本来就要几秒到十几秒，超时定成几秒，正常的长回答也会失败。P1-02 和 P1-04 会实测几十个真实回答的耗时，到时再改；P3 给整个请求加总时限以后，单次调用也不会超过剩下的时间。
- 重试用 openai SDK 自己的：两次之间等 0.5 秒、1 秒、2 秒、4 秒，最多 8 秒，每次随机少等 0 到 25%。SDK 默认会按网关的 `Retry-After` 最多等 120 秒。传输层看到 `Retry-After` 超过这套配置的上限，就在响应头里加 `x-should-retry: false`，SDK 看到这个头不再重试，直接报错。和 P0-05 一样，网关要求 online 调用等 30 秒，就直接失败，不等。
- 还考虑过：像 P0-05 那样在传输层自己写重试，让 SDK 不重试。好处是能设每次等多久和总时限。没选：要再写一份和 P0-05 差不多的重试代码；SDK 的等待本来就短（最多 8 秒），唯一的问题是 `Retry-After` 可能很长，加一个响应头就解决了。

### 2026-10-17 P1-01：对话模型的价格仍是估计值
网关的模型列表不带价格，每次调用的响应里也没有费用，这次查不到 deepseek-v4-flash 的实际扣费，价格不改：仍按输入 2 元、输出 8 元每百万 token 算。`ModelPrice` 加了 `estimated` 字段，这个模型写成 `true`，账本里对话调用的每一行都带 `price_estimated: true`，以后知道真实价格只改配置。

这次 `llm` 测试一共 4 次对话调用，按估计价花了 0.002102 元，账本总额从 0.000040 元变成 0.002142 元。`json_schema` 那次返回 400，账本里没有写这一次。

### 2026-10-17 P1-01：网关不可用或预算用完时，手动设 CHAT_BACKEND=fake，程序不自动切换
设 `CHAT_BACKEND=fake` 以后，三套配置都返回 `FakeChatModel`，每次都回答 `chat_fake_reply` 这句固定的话（默认 `[fake chat model] no answer`）：不发请求，不写账本，也不检查网关地址和价格。假模型上的结构化输出拿不到对象（固定回答里没有工具调用），会抛 `StructuredOutputError`。`FakeChatModel` 为此加了 `default_response`：脚本里的回答用完以后用它，不设时仍然报错。

和 P0-05 的向量一样，不做「网关出错就自动换成假模型」：自动换了以后，评测分数和返回给用户的回答里会混进假回答，程序却不报错。网关出错时按设计 16.8 报 `llm_timeout` 或 `llm_error`，由调用方决定只返回检索到的段。预算用完时 `BudgetExceeded` 的错误信息里提示设 `EMBED_BACKEND=fake` 和 `CHAT_BACKEND=fake`。

### 2026-10-17 P1-01：对话模型调用的四个错误类别
`ErrorCategory` 加了四个类别，`chat_error_category(exc)` 把调用时抛出的异常换成类别：

| 类别 | 什么情况 |
|---|---|
| `llm_timeout` | 连接或读取超时，重试已经用完 |
| `llm_error` | 其他网关错误：连不上、4xx、429、5xx |
| `budget_exceeded` | 预算检查不让发，请求没有发出去 |
| `structured_output_failed` | 回答转不成要求的 Pydantic 对象 |

不属于这四类的异常返回 None，免得把程序自己的 bug 当成网关故障。设计 16.8 里「写回答」这一行只说失败时只返回检索到的段，没给类别名。还考虑过只用一个 `llm_error`：调用方就看不出是网关慢了，还是预算用完了；预算用完要换假模型或者等主人加预算，处理办法不一样，所以分开。
