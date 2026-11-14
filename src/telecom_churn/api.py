import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from telecom_churn.modeling import load_model
from telecom_churn.schema import Prediction, PredictRequest
from telecom_churn.scoring import score_customers

# 获取本模块的日志记录器，使用服务统一的日志配置。
logger = logging.getLogger("telecom_churn.api")
# STATIC：与 Python 包一起发布的前端资源目录。
STATIC = Path(__file__).parent / "static"


# 创建应用并注册生命周期、鉴权、中间件、业务路由与静态页面。
def create_app(model_path: Path | None = None, api_key: str | None = None) -> FastAPI:
    # 计算并保存 path；按右侧表达式完成取列、计算或结果构造。
    path = model_path or Path(os.getenv("CHURN_MODEL_PATH", "artifacts"))
    # expected_key：显式传入或从环境变量读取的 API 密钥。
    expected_key = api_key if api_key is not None else os.getenv("CHURN_API_KEY", "")
    # environment：当前服务运行环境。
    environment = os.getenv("CHURN_ENV", "development")
    # 检查 environment == 'production' and (not expected_key)；满足条件才执行下面的分支。
    if environment == "production" and not expected_key:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("CHURN_API_KEY is required in production")

    # 进程启动时载入一次模型；退出时释放应用持有的引用。
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 清空应用持有的模型引用，未加载成功时保持不可用状态。
        app.state.bundle = None
        # 执行可能失败的操作，并在下方捕获预期异常。
        try:
            # 载入并校验模型，将其缓存到进程内供所有请求共用。
            app.state.bundle = load_model(path)
            # 输出运行日志以便追踪服务状态。
            logger.info("model_loaded version=%s", app.state.bundle["metadata"]["model_version"])
        except FileNotFoundError, ValueError, KeyError:
            # 记录加载失败的异常栈；进程仍可提供健康检查，模型接口返回未就绪。
            logger.exception("model_load_failed")
        # 启动阶段到此结束，把控制权交给服务器；退出时从 yield 后继续清理。
        yield
        # 清空应用持有的模型引用，未加载成功时保持不可用状态。
        app.state.bundle = None

    # 计算并保存 app；调用 FastAPI 完成本步处理。
    app = FastAPI(title="Telecom Churn Prediction", version="0.1.0", lifespan=lifespan)

    # 配置了密钥时执行恒定时间比较，校验失败返回 401。
    def authenticate(x_api_key: str | None = Header(default=None)):
        # 检查 expected_key and (not secrets.compare_digest(x_api_key or '', expected_key))；满足条件才执行下面的分支。
        if expected_key and not secrets.compare_digest(x_api_key or "", expected_key):
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise HTTPException(status_code=401, detail="Invalid API key")

    # 取得已载入的模型；未就绪时返回 503。
    def get_bundle():
        # bundle：模型管道、元数据和训练参考分布组成的工件包。
        bundle = getattr(app.state, "bundle", None)
        # 检查 bundle is None；满足条件才执行下面的分支。
        if bundle is None:
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise HTTPException(status_code=503, detail="Model unavailable")
        # 把就绪模型提供给依赖它的路由。
        return bundle

    # 为请求附加追踪编号，记录状态码与耗时，不记录客户特征正文。
    @app.middleware("http")
    async def request_metrics(request: Request, call_next):
        # request_id：用于关联本次请求日志的随机编号。
        request_id = uuid4().hex
        # started：请求开始时的单调时钟值。
        started = time.perf_counter()
        # response：下游处理返回的 HTTP 响应。
        response = await call_next(request)
        # 把追踪编号放进响应头，便于将客户端问题与服务日志对应。
        response.headers["X-Request-ID"] = request_id
        # 禁止浏览器根据内容猜测 MIME 类型，按声明类型解释响应。
        response.headers["X-Content-Type-Options"] = "nosniff"
        # 输出运行日志以便追踪服务状态。
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        # 返回原业务响应，保留额外加入的追踪及类型保护响应头。
        return response

    # 返回进程存活状态，不代表模型已经就绪。
    @app.get("/health")
    def health():
        # 返回进程存活标记，不尝试加载或使用模型。
        return {"status": "ok"}

    # 只有模型加载成功才返回就绪及版本信息。
    @app.get("/ready")
    def ready(bundle=Depends(get_bundle)):
        # 返回就绪标记及已加载模型版本。
        return {"status": "ready", "model_version": bundle["metadata"]["model_version"]}

    # 返回当前模型版本、验证比较、测试指标和特征重要性。
    @app.get("/api/v1/model", dependencies=[Depends(authenticate)])
    def model_info(bundle=Depends(get_bundle)):
        # 返回模型元数据供页面展示算法、指标和特征重要性。
        return bundle["metadata"]

    # 把经过 Pydantic 校验的客户列表转换为表格并复用批量评分逻辑。
    @app.post(
        "/api/v1/predict", response_model=list[Prediction], dependencies=[Depends(authenticate)]
    )
    def predict(payload: PredictRequest, bundle=Depends(get_bundle)):
        # 计算并保存 frame；调用 pd.DataFrame 完成本步处理。
        frame = pd.DataFrame([row.model_dump() for row in payload.customers])
        # 执行可能失败的操作，并在下方捕获预期异常。
        try:
            # 调用共用评分函数，并把表格转成可序列化的逐客户字典列表。
            return score_customers(frame, bundle).to_dict("records")
        except ValueError as exc:
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # 返回工作台首页，由同一服务提供静态资源。
    @app.get("/", include_in_schema=False)
    def index():
        # 发送包内首页文件。
        return FileResponse(STATIC / "index.html")

    # 注册静态资源路径，让浏览器能加载页面样式和脚本。
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    # 返回已配置路由和生命周期的 FastAPI 应用。
    return app
