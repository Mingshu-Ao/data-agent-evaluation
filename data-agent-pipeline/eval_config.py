"""评测 Pipeline 配置（不修改原 LLM_config.py）

密钥一律从环境变量读取；本文件已提交到 Git，仓库内不保存任何真实 Key。
运行前请设置环境变量（任设一个即可同时作为 Commander/Worker 的 Key）：

  DEEPSEEK_COMMANDER_API_KEY / DEEPSEEK_WORKER_API_KEY
  DEEPSEEK_API_BASE_URL（可选，默认 https://api.deepseek.com）

未设置时相关 LLM 调用会因缺少 Key 直接报错（这是预期行为，避免误用）。
"""
import os

_DEFAULT_BASE_URL = "https://api.deepseek.com"

# 仓库内不存放真实 Key；仅从环境变量注入。两者任设其一即可通用。
_COMMANDER_KEY = os.environ.get("DEEPSEEK_COMMANDER_API_KEY", "")
_WORKER_KEY = os.environ.get("DEEPSEEK_WORKER_API_KEY", "")
_FALLBACK_KEY = _COMMANDER_KEY or _WORKER_KEY

# Commander 用强推理模型做规划
COMMANDER_MODEL = "deepseek/deepseek-reasoner"
COMMANDER_BASE_URL = os.environ.get("DEEPSEEK_API_BASE_URL", _DEFAULT_BASE_URL)
COMMANDER_API_KEY = _COMMANDER_KEY or _FALLBACK_KEY

# Workers 用快速模型做执行
WORKER_MODEL = "deepseek/deepseek-chat"
WORKER_BASE_URL = os.environ.get("DEEPSEEK_API_BASE_URL", _DEFAULT_BASE_URL)
WORKER_API_KEY = _WORKER_KEY or _FALLBACK_KEY

# 备选 API（请通过环境变量注入，勿把真实 Key 写进文件）
# GLM-5.2: model="glm-5.2", base_url="https://open.bigmodel.cn/api/paas/v4"
#   （Key 通过环境变量 GLM_API_KEY 注入）
