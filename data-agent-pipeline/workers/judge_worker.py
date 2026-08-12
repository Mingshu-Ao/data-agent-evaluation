"""LLM 语义评分 Worker：用于 KramaBench/LakeQA/FDAbench 的自由文本/数值答案。

ScoreFlow（workers/score_worker.py）是确定性行相等评分，对单 cell 的自由文本
答案只能得 0；JudgeFlow 用 LLM 按 answer_type 语义判断正确性，输出
final_score ∈ [0,1]，喂给 eval_pipeline 的 0.99 阈值判定 correct。

结构照抄 answer_worker.py：类属性 llm + to_param()/set_para()/@start() 契约，
_kickoff() 返回 @start() 方法的返回值。
"""
from __future__ import annotations

import json
import re
from crewai import LLM
from crewai.flow import start
from pydantic import BaseModel
from TAS.Crewai.utils import AgentFlow
from eval_config import WORKER_MODEL, WORKER_BASE_URL, WORKER_API_KEY


class JudgeState(BaseModel):
    question: str = ""
    gold: list = []            # 标准答案表格（EvalTask.gold_answer，list[list]）
    prediction: list = []      # 预测答案表格（pred，list[list]）
    answer_type: str = ""      # numeric_exact / string_approximate / ... 空则按通用规则
    final_score: float = 0.0
    reason: str = ""


JUDGE_SYSTEM = """你是数据 Agent benchmark 的严格但公正的评分员。给定一个问题、参考答案和 Agent 的预测答案，
判断预测是否应判定为正确。只输出一个 JSON 对象，不要输出任何其他文本：
{"score": 0.0, "reason": "一句话理由"}

评分规则（按 answer_type 分支）：
- numeric_exact：预测与参考答案数值完全相等（容忍格式差异：千分位逗号、尾随 .0、量纲后缀）。
- numeric_approximate：预测数值在合理误差内接近参考答案。
- string_exact：忽略大小写与首尾空白后完全一致。
- string_approximate：语义等价即正确（同义表达、换措辞、数字与文字互转）。
- list_exact：两个列表逐元素一致（顺序敏感）。
- list_approximate：两个列表是同一集合（顺序无关）。
- 其他/未知：按常识判断是否表达了同一答案。

score 取值：完全正确 1.0；完全错误 0.0；部分正确按比例给 (0.0, 1.0)。
预测为空、无关或明显编造时 score=0.0。数值答案若预测含推导过程但最终数值正确，按正确计。"""


class JudgeFlow(AgentFlow[JudgeState]):
    name: str = "SemanticJudge"
    description: str = "用 LLM 对自由文本/数值答案做语义评分。"
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
                        "question": {"type": "string", "description": "任务问题"},
                        "gold": {"type": "array", "description": "标准答案表格，第一行是表头"},
                        "prediction": {"type": "array", "description": "预测答案表格，第一行是表头"},
                    },
                    "required": ["question", "gold", "prediction"],
                },
            },
        }

    def set_para(self, para: dict):
        self.state.question = str(para.get("question", ""))
        self.state.gold = para.get("gold") or []
        self.state.prediction = para.get("prediction") or []
        self.state.answer_type = str(para.get("answer_type", ""))

    @staticmethod
    def _extract_cell(table) -> str:
        """从 [header, [row]] 表格提取首个数据 cell，解一层嵌套 list / "[...]" 字面量。

        Krama/LakeQA/FDAbench 的 gold 是 [["answer"], [[str]]]，嵌套的 [[str]] 经
        csv 序列化后 cell 内容形如 "['2020']"，都需要解一层。
        """
        if not table:
            return ""
        rows = table[1:] if len(table) > 1 else table
        if not rows or not rows[0]:
            return ""
        cell = rows[0][0]
        if isinstance(cell, list):
            cell = cell[0] if cell else ""
        cell = str(cell).strip()
        if cell.startswith("[") and cell.endswith("]"):
            try:
                parsed = json.loads(cell)
                if isinstance(parsed, list) and parsed:
                    cell = str(parsed[0]).strip()
            except (json.JSONDecodeError, TypeError):
                pass
        return cell

    @staticmethod
    def _extract_json(response: str) -> dict:
        """从 LLM 响应中稳健地提取 JSON 对象（与 answer_worker 相同实现）。

        优先取 ```json ... ``` 围栏；否则取第一个平衡的 JSON 对象。
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
        start_i = text.find("{")
        if start_i != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start_i, len(text)):
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
                            return json.loads(text[start_i:i + 1])
                        except json.JSONDecodeError:
                            break
        return {}

    @start()
    def judge(self):
        gold = self._extract_cell(self.state.gold)
        pred = self._extract_cell(self.state.prediction)
        if not gold or not pred:
            self.state.final_score = 0.0
            self.state.reason = "empty gold/prediction"
            return {"final_score": 0.0, "reason": self.state.reason,
                    "gold": gold, "prediction": pred}

        prompt = (
            JUDGE_SYSTEM + "\n\n"
            f"Question:\n{self.state.question}\n\n"
            f"Expected answer:\n{gold}\n\n"
            f"Agent prediction:\n{pred}\n\n"
            f"answer_type: {self.state.answer_type or 'unknown'}"
        )
        try:
            raw = self.llm.call(prompt)
            parsed = self._extract_json(raw)
            try:
                score = float(parsed.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(1.0, score))
            reason = str(parsed.get("reason", "")) or "no reason"
        except Exception as e:  # noqa: BLE001 —— LLM 调用失败兜底为 0
            score = 0.0
            reason = f"judge call failed: {e}"
        self.state.final_score = score
        self.state.reason = reason
        return {"final_score": score, "reason": reason,
                "gold": gold, "prediction": pred, "answer_type": self.state.answer_type}
