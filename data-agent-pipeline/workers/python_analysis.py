"""Python 数据分析 Worker"""
import sys
import subprocess
from pathlib import Path
from crewai import LLM
from crewai.flow import start
from pydantic import BaseModel
from TAS.Crewai.utils import AgentFlow
from eval_config import WORKER_MODEL, WORKER_BASE_URL, WORKER_API_KEY


class PythonAnalysisState(BaseModel):
    context_dir: str = ""
    user_request: str = ""
    code: str = ""
    context_preview: str = ""  # 结构化查询预览（可选，帮助生成分析代码）
    result: dict = {}


class PythonAnalysisFlow(AgentFlow[PythonAnalysisState]):
    name: str = "PythonAnalysis"
    description: str = (
        "在指定数据目录下执行 Python 代码进行数据分析。pandas、numpy 已预装。"
    )
    llm = LLM(model=WORKER_MODEL, base_url=WORKER_BASE_URL, api_key=WORKER_API_KEY)

    def to_param(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context_dir": {"type": "string", "description": "数据目录路径"},
                        "user_request": {"type": "string", "description": "分析需求描述"},
                        "code": {"type": "string", "description": "可选：直接提供 Python 代码"},
                    },
                    "required": ["context_dir", "user_request"],
                },
            },
        }

    def set_para(self, para: dict):
        self.state.context_dir = para.get("context_dir", "")
        self.state.user_request = para.get("user_request", "")
        self.state.code = para.get("code", "")
        self.state.context_preview = para.get("context_preview", "")

    @start()
    def run_analysis(self):
        code = self.state.code
        if not code:
            prompt = (
                f"数据目录下有以下文件，请根据需求生成 Python 代码。\n"
                f"需求：{self.state.user_request}\n"
                f"工作目录：{self.state.context_dir}\n"
            )
            if self.state.context_preview:
                prompt += (
                    f"已探明的数据结构（来自结构化查询，仅供参考）：\n"
                    f"{self.state.context_preview[:3000]}\n"
                )
            prompt += (
                f"请生成可直接执行的 Python 代码（pandas 已可用），用 print() 输出结果。"
                f"只输出代码，不要解释。"
            )
            code = self.llm.call(prompt)
            # 去掉 LLM 可能加的 markdown fence
            import re
            m = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', code, re.DOTALL)
            if m:
                code = m.group(1).strip()

        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=self.state.context_dir,
                capture_output=True, text=True, timeout=30,
            )
            self.state.result = {
                "success": result.returncode == 0,
                "output": result.stdout[:5000],
                "stderr": result.stderr[:2000],
            }
        except subprocess.TimeoutExpired:
            self.state.result = {"success": False, "error": "执行超时（30秒）"}
        except Exception as e:
            self.state.result = {"success": False, "error": str(e)}

        return self.state.result
