"""评分 Worker：对比预测答案和标准答案"""
from crewai.flow import start
from pydantic import BaseModel
from TAS.Crewai.utils import AgentFlow


class ScoreState(BaseModel):
    prediction: list = []
    gold: list = []
    recall: float = 0.0
    redundancy_penalty: float = 0.0
    final_score: float = 0.0


class ScoreFlow(AgentFlow[ScoreState]):
    name: str = "Score"
    description: str = "对比预测答案和标准答案，计算 Recall、Redundancy Penalty、Final Score。"

    def to_param(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prediction": {"type": "array", "description": "预测答案表格，第一行是表头"},
                        "gold": {"type": "array", "description": "标准答案表格，第一行是表头"},
                    },
                    "required": ["prediction", "gold"],
                },
            },
        }

    def set_para(self, para: dict):
        self.state.prediction = para.get("prediction", [])
        self.state.gold = para.get("gold", [])

    @staticmethod
    def _safe_eq(a, b) -> bool:
        return str(a).strip().lower() == str(b).strip().lower()

    @start()
    def compute_score(self):
        gold = self.state.gold
        pred = self.state.prediction
        if not gold or not pred:
            self.state.final_score = 0.0
            return {"recall": 0, "redundancy_penalty": 0, "final_score": 0}

        gold_data = gold[1:] if len(gold) > 1 else gold
        pred_data = pred[1:] if len(pred) > 1 else pred

        matches = sum(
            1 for g in gold_data for p in pred_data
            if len(g) == len(p) and all(self._safe_eq(a, b) for a, b in zip(g, p))
        )
        recall = matches / len(gold_data) if gold_data else 0

        if len(pred_data) > len(gold_data) and gold_data:
            penalty = min((len(pred_data) - len(gold_data)) / len(gold_data) * 0.5, 0.5)
        else:
            penalty = 0

        final = max(0, recall - penalty)
        self.state.recall = recall
        self.state.redundancy_penalty = penalty
        self.state.final_score = final

        return {
            "recall": round(recall, 4),
            "redundancy_penalty": round(penalty, 4),
            "final_score": round(final, 4),
            "gold_rows": len(gold_data),
            "pred_rows": len(pred_data),
            "matched": matches,
        }
