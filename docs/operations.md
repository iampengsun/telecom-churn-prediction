# 部署与运维

## 服务配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| CHURN_MODEL_PATH | artifacts | 模型根目录、版本目录或 joblib 文件 |
| CHURN_API_KEY | 空 | 模型信息和预测接口的访问密钥 |
| CHURN_ENV | development | production 环境强制要求密钥 |

本地服务默认监听 `127.0.0.1:8000`。API Key 由请求头 `X-API-Key` 提供。工作台仅在当前页面内存中使用密钥，不写入浏览器存储。

## 容器部署

```bash
docker compose build
docker compose up -d
```

Compose 从环境或 `.env` 读取 `CHURN_API_KEY`，挂载本地模型目录为只读。容器以非 root 用户运行，文件系统只读，`/tmp` 为临时内存盘，端口绑定至宿主机回环地址。

容器运行 Python 3.12。模型由容器训练可保持训练与服务环境完全一致：

```bash
docker compose run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/artifacts:/app/trained" \
  api churn train --features /app/data/features.csv --artifacts /app/trained
```

目录权限必须允许容器 UID 10001 读写对应训练目录。模型加载依赖一致的 scikit-learn、LightGBM、NumPy 和 Joblib 版本，依赖变更后应重新训练和验证。

## 发布与回滚

模型根目录中的 `latest.json` 指向当前版本。训练成功后更新该指针；API 服务不会热加载。发布时重启服务并检查 `/ready` 返回的 `model_version`。回滚使用 `CHURN_MODEL_PATH=artifacts/<version>` 指定已有版本，再重启工作进程。

`/health` 只检查进程；`/ready` 检查是否成功加载模型。模型缺失、校验失败或特征版本不匹配时，预测接口返回 503。启动日志记录具体原因。

## 日志与访问边界

请求日志包含请求编号、方法、路径、状态和耗时；不记录特征请求体、客户号码或密钥。输入客户编号应使用业务脱敏标识。

公网或跨网络部署需由网关提供 TLS、访问控制、请求体限制和限流。预测接口限制单次 1,000 条客户记录；应用未内置多租户隔离、持久化审计库或CRM写入连接器。

## 批处理作业

业务调度依次执行数据水位检查、特征构建、评分、资格筛选、名单输出和漂移检查。CLI 各阶段输出文件路径；异常返回非零状态。名单文件按评分日和模型版本归档，避免覆盖尚未消费的名单。

漂移报告告警不自动中断评分；调度系统可按 `alert_count` 及业务规则控制名单发布。延迟模型评估使用成熟标签特征表执行：

```bash
churn evaluate --features data/matured-features.csv --output reports/evaluation.json
```

训练集评估不作为上线效果。月度历史评估与实际触达结果应分别存储。
