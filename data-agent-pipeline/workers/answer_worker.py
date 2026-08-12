"""答案生成 Worker"""
import json
import re
from crewai import LLM
from crewai.flow import start
from pydantic import BaseModel
from TAS.Crewai.utils import AgentFlow
from eval_config import WORKER_MODEL, WORKER_BASE_URL, WORKER_API_KEY


class AnswerState(BaseModel):
    task_question: str = ""
    analysis_result: str = ""
    knowledge: str = ""  # 评测知识库（eval_knowledge.json），可选
    answer: dict = {}


class AnswerFlow(AgentFlow[AnswerState]):
    name: str = "Answer"
    description: str = "根据任务问题和分析结果生成最终答案表格。"
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
                        "task_question": {"type": "string", "description": "任务问题"},
                        "analysis_result": {"type": "string", "description": "分析结果"},
                    },
                    "required": ["task_question", "analysis_result"],
                },
            },
        }

    def set_para(self, para: dict):
        self.state.task_question = para.get("task_question", "")
        self.state.analysis_result = str(para.get("analysis_result", ""))
        self.state.knowledge = str(para.get("knowledge", ""))

    @start()
    def generate_answer(self):
        prompt = (
            f"Task: {self.state.task_question}\n\n"
            f"Analysis results:\n{self.state.analysis_result}\n\n"
        )
        if self.state.knowledge:
            prompt += f"Evaluation knowledge:\n{self.state.knowledge}\n\n"
        prompt += (
            f"Based ONLY on the observed data above, output the final answer "
            f"as a JSON object with 'columns' and 'rows'. "
            f"NEVER guess or fabricate values.\n"
            f"Format: ```json\n{{\"columns\": [...], \"rows\": [[...]]}}\n```"
        )
        response = self.llm.call(prompt)
        self.state.answer = self._extract_json(response)
        return self.state.answer

    @staticmethod
    def _extract_json(response: str) -> dict:
        """从 LLM 响应中稳健地提取 JSON 对象。

        优先取 ```json ... ``` 围栏；否则取第一个平衡的 JSON 对象
        （避免贪婪匹配把尾随文本里的括号也吃进去）。
        """
        text = str(response).strip()
        try:
            m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if m:
                return json.loads(m.group(1).strip())
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # 平衡括号扫描：取第一个完整的 {...} 顶层对象
        start = text.find("{")
        if start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
        return {"columns": [], "rows": [], "error": "解析失败"}
