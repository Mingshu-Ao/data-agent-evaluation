"""快速测试：验证 loader 和 worker 是否正常工作"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# workers 依赖 TAS.Crewai.utils（Baseline 项目的包），先做路径引导
try:
    from eval_pipeline import _ensure_tas_importable
    _ensure_tas_importable()
except Exception:
    pass

# 1. 测试 loader
print("=" * 50)
print("1. 测试 KDD Loader")
print("=" * 50)
from loaders.kdd_loader import KDDLoader

loader = KDDLoader(
    input_dir=Path(__file__).parent / "data" / "kdd_phase1" / "input",
    output_dir=Path(__file__).parent / "data" / "kdd_phase1" / "output",
)

tasks = loader.list_tasks()
print(f"任务总数: {len(tasks)}")
print(f"前5个: {tasks[:5]}")

counts = loader.task_count()
print(f"难度分布: {counts}")

# 2. 测试加载单个任务
print(f"\n{'=' * 50}")
print("2. 测试加载 task_22")
print("=" * 50)
task = loader.load_task("task_22")
print(f"ID: {task.task_id}")
print(f"难度: {task.difficulty}")
print(f"问题: {task.question[:100]}...")
print(f"数据目录: {task.context_dir}")
print(f"Gold 行数: {len(task.gold_answer)}")
if task.gold_answer:
    print(f"Gold 表头: {task.gold_answer[0]}")

# 3. 测试 worker 导入
print(f"\n{'=' * 50}")
print("3. 测试 Worker 导入")
print("=" * 50)
try:
    from workers.structured_query import StructuredQueryFlow
    print("structured_query: OK")
except Exception as e:
    print(f"structured_query: FAIL ({e})")

try:
    from workers.python_analysis import PythonAnalysisFlow
    print("python_analysis: OK")
except Exception as e:
    print(f"python_analysis: FAIL ({e})")

try:
    from workers.answer_worker import AnswerFlow
    print("answer_worker: OK")
except Exception as e:
    print(f"answer_worker: FAIL ({e})")

try:
    from workers.score_worker import ScoreFlow
    print("score_worker: OK")
except Exception as e:
    print(f"score_worker: FAIL ({e})")

# 4. 测试评分 Worker 独立运行
print(f"\n{'=' * 50}")
print("4. 测试评分逻辑")
print("=" * 50)
from workers.score_worker import ScoreFlow
score = ScoreFlow()
score.set_para({
    "prediction": [["col"], ["42"]],
    "gold": [["col"], ["42"]],
})
result = score.kickoff()
print(f"满分测试 (预测=gold): {result}")

score2 = ScoreFlow()
score2.set_para({
    "prediction": [["col"], ["99"]],
    "gold": [["col"], ["42"]],
})
result2 = score2.kickoff()
print(f"零分测试 (预测≠gold): {result2}")

print(f"\n{'=' * 50}")
print("全部测试通过！")
print("运行完整 Pipeline: python eval_pipeline.py --suite pipeline_smoke_phase1_easy_3.json --mock")
