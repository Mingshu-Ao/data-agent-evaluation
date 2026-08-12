"""Baseline 适配器：通过 subprocess 调用 Baseline Agent 并收集输出

支持的 Agent：
  - react:             dabench run-benchmark
  - dagent-lite:       dabench run-benchmark-dagent
  - agenticdata-lite:  dabench run-benchmark-agenticdata
  - mini-aop:          dabench run-benchmark-mini-aop

对接方式（已按 PHASE_1 实际 CLI 核实）：
  - `dabench run-benchmark` 只接受 `--config` 和 `--limit`，**不接受 `--suite`**；
    任务集来自 `config.dataset.root_path` 目录下的 task_N/。
    因此 Pipeline 通过 prepare_task_dir() 把 suite 任务 stage 到 dataset/input/，
    再写临时 YAML 把 dataset.root_path / run.output_dir / run_id 注入给 dabench。
  - 若某个版本 CLI 支持 `--suite`，会动态检测并附加该参数（兼容 handoff 文档约定）。
  - 真实运行输出布局：`<output_dir>/baseline_runs/<run_id>/<task_id>/{prediction.csv, trace.json}`
    + summary.json（见 collect_task_outputs）。
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

AGENT_COMMANDS = {
    "react": "run-benchmark",
    "dagent-lite": "run-benchmark-dagent",
    "agenticdata-lite": "run-benchmark-agenticdata",
    "mini-aop": "run-benchmark-mini-aop",
}

# 协议 §5 定义的 9 类失败 + 2 个 Pipeline 哨兵
_CANONICAL_FAILURES = {
    "max_steps", "timeout", "model_api", "invalid_model_output", "tool_error",
    "missing_data", "preprocessing_error", "invalid_answer", "wrong_answer",
}
_SENTINELS = {"missing_trace", "unknown"}


_TASK_INT_RE = re.compile(r"^task_(\d+)$")
# 非 task_<int> ID 的映射起点：KDD 真实 ID 远小于 100000，映射空间足够
_MAPPED_ID_OFFSET = 100000


class TaskIdMapper:
    """把任意 benchmark 的 task_id 映射为 Baseline 能接受的 task_<int>。

    Baseline 的 `_task_number`（dataset.py）只认 `task_<int>` 目录名，且
    `get_task` 校验 `task.json.task_id == 目录名`。FDAbench（FDA0001）、
    Krama（legal-hard-1）、LakeQA（lakeqa-full:EQA...）的原始 ID 不能直接喂。

    对外（records / evaluation.json）始终用原始 task_id；只有 stage 目录名、
    task.json.task_id、tmp_suite.task_ids、collect 目录名用映射后的 ID。
    task_<int> 恒等；其余按 suite 内序号映射到 task_{100000+i}，带碰撞避让。

    用法:
        mapper = TaskIdMapper(task_ids)
        staged_id = mapper.to_baseline(tid)        # 原始 → task_<int>
        original  = mapper.from_baseline(staged_id)  # 回逆
    """

    def __init__(self, task_ids: list[str]):
        self._to: dict[str, str] = {}
        self._from: dict[str, str] = {}
        used = {int(m.group(1)) for tid in task_ids if (m := _TASK_INT_RE.match(tid))}
        next_free = _MAPPED_ID_OFFSET
        for tid in task_ids:
            if _TASK_INT_RE.match(tid):
                self._to[tid] = tid
                self._from[tid] = tid
                continue
            while next_free in used:
                next_free += 1
            mapped = f"task_{next_free}"
            used.add(next_free)
            next_free += 1
            self._to[tid] = mapped
            self._from[mapped] = tid

    def to_baseline(self, task_id: str) -> str:
        """原始 task_id → task_<int>（恒等或映射）。未知 ID 原样返回。"""
        return self._to.get(task_id, task_id)

    def from_baseline(self, mapped_id: str) -> str:
        """task_<int> → 原始 task_id（collect 后目录改名回填用）。"""
        return self._from.get(mapped_id, mapped_id)


class BaselineAdapter:
    """
    封装 Baseline Agent 的调用：
    1. 生成统一任务目录（prepare_task_dir）
    2. 探测 venv 里的 dabench 入口与本地 config
    3. 写临时 YAML 注入数据/输出路径并 subprocess 调用 dabench
    4. 从真实 run 输出目录收集 prediction.csv / trace.json
    """

    _cli_option_cache: dict = {}

    def __init__(
        self,
        project_dir: Path,
        venv_dir: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.project_dir = Path(project_dir)
        self._venv_dir_hint = venv_dir
        self._config_path_hint = config_path
        self._dabench = None
        self.config_path: Optional[Path] = None

    # ---------- 环境探测 ----------

    def _candidate_venv_names(self) -> list[str]:
        names = []
        if self._venv_dir_hint:
            names.append(self._venv_dir_hint)
        names += [".venv-dagent", ".venv", ".venv-dabench"]
        return names

    def _candidate_dabench_executables(self, venv: Path) -> list[Path]:
        """一个 venv 里 dabench 可能的入口（Windows Scripts/ 与 POSIX bin/，带/不带 .exe）"""
        scripts = venv / "Scripts"
        bin_dir = venv / "bin"
        candidates = []
        for d in (scripts, bin_dir):
            for name in ("dabench", "dabench.exe", "dabench.cmd", "dabench.bat"):
                candidates.append(d / name)
        return candidates

    def _resolve_dabench(self) -> Path:
        if self._dabench is not None:
            return self._dabench

        probed: list[Path] = []
        for name in self._candidate_venv_names():
            venv = self.project_dir / name
            for cand in self._candidate_dabench_executables(venv):
                if cand.exists():
                    self._dabench = cand
                    return cand
                probed.append(cand)

        # 兜底：扫描项目根下所有含 "venv" 的目录
        if self.project_dir.is_dir():
            for child in sorted(self.project_dir.iterdir()):
                if child.is_dir() and "venv" in child.name.lower():
                    for cand in self._candidate_dabench_executables(child):
                        if cand.exists():
                            self._dabench = cand
                            return cand

        raise FileNotFoundError(
            f"dabench 可执行文件未找到（已探测 {len(probed)} 个候选路径）。\n"
            f"请先在 {self.project_dir} 中安装: .\\.venv\\Scripts\\python.exe -m pip install -e .\n"
            f"或通过 --venv 指定 venv 目录名。"
        )

    def _candidate_configs(self) -> list[Path]:
        configs_dir = self.project_dir / "configs"
        names = []
        if self._config_path_hint:
            names.append(Path(self._config_path_hint))
            if not Path(self._config_path_hint).is_absolute():
                names.append(self.project_dir / self._config_path_hint)
        names += [
            configs_dir / "kdd_phase1_compare.local.yaml",   # handoff 文档默认
            configs_dir / "react_baseline.local.yaml",
            configs_dir / "react_baseline.yaml",
        ]
        if configs_dir.is_dir():
            # 任意 *local.yaml（排除 *.example.yaml），字母序保证稳定
            names += sorted(
                p for p in configs_dir.glob("*.local.yaml") if not p.name.endswith(".example.yaml")
            )
        seen, out = set(), []
        for p in names:
            r = p.resolve()
            if r not in seen:
                seen.add(r)
                out.append(p)
        return out

    def _resolve_config(self) -> Path:
        if self.config_path is not None:
            return self.config_path
        for cand in self._candidate_configs():
            if cand.is_file():
                self.config_path = cand
                return cand
        raise FileNotFoundError(
            f"未找到 Baseline 本地 config（探测 {self.project_dir / 'configs'} 下 *.yaml / *.local.yaml，"
            f"排除 *.example.yaml）。请通过 --config 指定。"
        )

    def _cli_supports_option(self, option: str) -> bool:
        """探测 CLI 是否支持某选项（例如 --suite）。结果缓存。"""
        key = (str(self._resolve_dabench()), option)
        if key in self._cli_option_cache:
            return self._cli_option_cache[key]

        # 用任意一个子命令的 --help 判断（选项按子命令注册；这里用 run-benchmark）
        subcommand = AGENT_COMMANDS["react"]
        try:
            result = subprocess.run(
                [str(self._resolve_dabench()), subcommand, "--help"],
                cwd=str(self.project_dir), capture_output=True, text=True, timeout=60,
            )
            text = (result.stdout or "") + (result.stderr or "")
            supported = option in text
        except Exception:
            supported = False
        self._cli_option_cache[key] = supported
        return supported

    def _check_env(self):
        """检查环境是否可用（探测 dabench 与 config）"""
        self._resolve_dabench()
        self._resolve_config()

    # ---------- 临时 config ----------

    @staticmethod
    def _load_agent_block(config_path: Path) -> dict:
        """读取基础 config 的 agent 块（模型、密钥等）。避免依赖 pyyaml，做轻量解析。"""
        try:
            import yaml
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except ImportError:
            payload = BaselineAdapter._parse_yaml_simple(config_path)
        agent = payload.get("agent") or {}
        return {
            "model": str(agent.get("model", "")),
            "api_base": str(agent.get("api_base", "")),
            "api_key": str(agent.get("api_key", "")),
            "max_steps": int(agent.get("max_steps", 16)),
            "temperature": float(agent.get("temperature", 0.0)),
        }

    @staticmethod
    def _parse_yaml_simple(path: Path) -> dict:
        """极简 YAML 解析（仅本项目 config 结构），用于无 pyyaml 的环境。"""
        root: dict = {}
        section: Optional[str] = None
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not line.startswith(" ") and line.rstrip().endswith(":"):
                section = line.rstrip()[:-1].strip()
                root[section] = {}
            elif ":" in stripped and section is not None:
                k, _, v = stripped.partition(":")
                root[section][k.strip()] = v.strip().strip("'\"") if v.strip() else ""
        return root

    @staticmethod
    def _dump_config_yaml(cfg: dict) -> str:
        """把注入后的 config 序列化为 YAML（固定结构，无需 pyyaml）"""
        lines = ["# Generated by data_agent_pipeline baseline_adapter; safe to delete.", "dataset:"]
        lines.append(f"  root_path: {json.dumps(cfg['dataset']['root_path'])}")
        agent = cfg["agent"]
        lines.append("agent:")
        for key in ("model", "api_base", "api_key"):
            lines.append(f"  {key}: {json.dumps(str(agent.get(key, '')))}")
        lines.append(f"  max_steps: {int(agent.get('max_steps', 16))}")
        lines.append(f"  temperature: {float(agent.get('temperature', 0.0))}")
        run = cfg["run"]
        lines.append("run:")
        lines.append(f"  output_dir: {json.dumps(str(run['output_dir']))}")
        lines.append(f"  run_id: {json.dumps(str(run['run_id']))}")
        lines.append(f"  max_workers: {int(run.get('max_workers', 1))}")
        lines.append(f"  task_timeout_seconds: {int(run.get('task_timeout_seconds', 600))}")
        return "\n".join(lines) + "\n"

    # ---------- 主流程 ----------

    def run_agent(
        self,
        agent: str,
        dataset_dir: Path,
        suite_path: Path,
        output_dir: Path,
        task_timeout: int = 600,
        max_workers: int = 1,
        run_id: Optional[str] = None,
    ) -> dict:
        """
        运行 Baseline Agent。

        流程：
          1. prepare_task_dir() 已把 suite 任务 stage 到 dataset_dir/input/task_N/。
          2. 写临时 YAML：dataset.root_path=dataset_dir/input，run.output_dir=output_dir/baseline_runs，
             run_id=新时间戳，max_workers=1，task_timeout_seconds=task_timeout。
          3. 调 `dabench <cmd> --config <tmp_config>`（若 CLI 支持 --suite 则追加）。
          4. 真实输出落在 <output_dir>/baseline_runs/<run_id>/<task_id>/。

        Args:
            agent: "react" | "dagent-lite" | "agenticdata-lite" | "mini-aop"
            dataset_dir: 包含 input/（和 output/）的目录
            suite_path: suite JSON（仅当 CLI 支持 --suite 时传入）
            output_dir: 输出根目录（run 输出在其 baseline_runs/<run_id> 下）
            task_timeout: 单任务超时秒数（写入 config 由 Baseline 强制）
            max_workers: 并发数（联调建议 1）
            run_id: 可选 run 目录名；缺省生成 pipeline_<时间戳>

        Returns:
            {"success": bool, "run_id": str, "run_dir": str, "config_path": str,
             "stdout": str, "stderr": str, "error": str | None}
        """
        cmd = AGENT_COMMANDS.get(agent)
        if not cmd:
            return {"success": False, "error": f"未知 agent: {agent}", "run_id": run_id or "", "run_dir": "", "config_path": "", "stdout": "", "stderr": ""}

        self._check_env()
        dabench = self._resolve_dabench()
        base_config = self._resolve_config()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_root = (Path(dataset_dir) / "input").resolve()
        if not input_root.is_dir():
            return {"success": False, "error": f"dataset 输入目录不存在: {input_root}", "run_id": run_id or "", "run_dir": "", "config_path": "", "stdout": "", "stderr": ""}

        run_id = run_id or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        run_output_root = (output_dir / "baseline_runs").resolve()

        cfg = {
            "dataset": {"root_path": str(input_root)},
            "agent": self._load_agent_block(base_config),
            "run": {
                "output_dir": str(run_output_root),
                "run_id": run_id,
                "max_workers": max_workers,
                "task_timeout_seconds": task_timeout,
            },
        }
        if not cfg["agent"].get("model"):
            return {"success": False, "error": f"基础 config 缺少 agent.model: {base_config}", "run_id": run_id, "run_dir": "", "config_path": "", "stdout": "", "stderr": ""}

        tmp_dir = output_dir / "tmp_configs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_config = tmp_dir / f"{run_id}.yaml"
        tmp_config.write_text(self._dump_config_yaml(cfg), encoding="utf-8")

        args = [str(dabench), cmd, "--config", str(tmp_config)]
        if self._cli_supports_option("--suite"):
            args += ["--suite", str(suite_path)]

        try:
            result = subprocess.run(
                args,
                cwd=str(self.project_dir),
                capture_output=True, text=True,
                timeout=task_timeout * 20,  # 整体兜底超时（含 max_workers 并行）
            )
            run_dir = run_output_root / run_id
            return {
                "success": result.returncode == 0,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "config_path": str(tmp_config),
                "model": str(cfg["agent"].get("model", "")),
                "stdout": result.stdout[-3000:],
                "stderr": result.stderr[-1000:],
                "error": None if result.returncode == 0 else result.stderr[-1000:] or "dabench 非零退出",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Baseline agent 整体超时", "run_id": run_id, "run_dir": str(run_output_root / run_id), "config_path": str(tmp_config), "stdout": "", "stderr": ""}
        except Exception as e:
            return {"success": False, "error": str(e), "run_id": run_id, "run_dir": str(run_output_root / run_id), "config_path": str(tmp_config), "stdout": "", "stderr": ""}

    @staticmethod
    def collect_task_outputs(run_dir: Path, target_dir: Path, task_ids: list[str],
                             id_map: Optional[dict[str, str]] = None) -> dict:
        """
        把真实 run 输出 <run_dir>/<task_id>/{prediction.csv, trace.json} 收集到
        Pipeline 统一布局 <target_dir>/<task_id>/。返回 {task_id: {"prediction": bool, "trace": bool}}。

        id_map: {baseline_id: original_id}，把映射后的 task_<int> 目录改名回原始 ID
        （用于 FDAbench/Krama/LakeQA）。默认 None 恒等。
        """
        run_dir = Path(run_dir)
        target_dir = Path(target_dir)
        collected = {}
        for tid in task_ids:
            src = run_dir / tid
            dst = target_dir / (id_map.get(tid, tid) if id_map else tid)
            dst.mkdir(parents=True, exist_ok=True)
            rec = {"prediction": False, "trace": False}
            for name in ("prediction.csv", "trace.json"):
                f = src / name
                if f.exists():
                    shutil.copy2(f, dst / name)
                    rec[name.split(".")[0]] = True
            collected[tid] = rec
        return collected

    @staticmethod
    def parse_summary(run_dir: Path) -> Optional[dict]:
        """解析 dabench 写入的 summary.json（任务级运行结果摘要）"""
        p = Path(run_dir) / "summary.json"
        if not p.exists():
            return None
        with p.open(encoding="utf-8") as f:
            return json.load(f)

    # ---------- 输出解析 ----------

    @staticmethod
    def parse_prediction(task_dir: Path) -> Optional[list]:
        """解析 prediction.csv，返回 [header, row1, row2, ...]"""
        pred_path = task_dir / "prediction.csv"
        if not pred_path.exists():
            return None
        with pred_path.open(encoding="utf-8") as f:
            return list(csv.reader(f))

    @staticmethod
    def parse_trace(task_dir: Path) -> Optional[dict]:
        """解析 trace.json"""
        trace_path = task_dir / "trace.json"
        if not trace_path.exists():
            return None
        with trace_path.open(encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def classify_failure(trace: Optional[dict], timeout: int = 600) -> str:
        """
        根据 trace 和 timeout 分类失败原因（协议 §5 的 9 类）。
        另有两个 Pipeline 哨兵：missing_trace（无 trace）、unknown（无法归类）。

        返回值集合 = 9 类协议失败 + 2 个哨兵。
        """
        if not trace or not isinstance(trace, dict):
            return "missing_trace"

        failure = str(trace.get("failure_reason") or "")
        steps = trace.get("steps") or []
        answer = trace.get("answer")
        answer_is_dict = isinstance(answer, dict)
        elapsed = trace.get("e2e_elapsed_seconds")
        try:
            elapsed = float(elapsed)
        except (TypeError, ValueError):
            elapsed = 0.0

        lowered = failure.lower()
        if "missing" in lowered or "not found" in lowered:
            return "missing_data"
        if "preprocess" in lowered or "ocr" in lowered or "asr" in lowered:
            return "preprocessing_error"
        if "max_steps" in lowered:
            return "max_steps"
        if timeout and elapsed >= timeout * 0.95:
            return "timeout"
        if "model" in lowered or "api" in lowered or "rate" in lowered:
            return "model_api"
        if any(s.get("action") == "__error__" for s in steps if isinstance(s, dict)):
            return "invalid_model_output"
        if any(not s.get("ok") for s in steps if isinstance(s, dict) and s.get("action") not in ("__error__", "__plan__")):
            return "tool_error"

        # 已提交但结构不合法 / 与 gold 不匹配
        if answer_is_dict:
            if not answer.get("rows"):
                return "invalid_answer"
            return "wrong_answer"
        if answer is None:
            return "max_steps"
        # answer 存在但不是 dict（异常格式）
        return "invalid_answer"

    @staticmethod
    def prepare_task_dir(loader, task_id: str, target_dir: Path, use_symlink: bool = False,
                         baseline_id: Optional[str] = None):
        """
        将原始数据放到统一任务目录格式。
        目录结构: target_dir/input/task_id/ 和 target_dir/output/task_id/

        Args:
            loader: 任意 benchmark loader（统一 list_tasks/load_task 接口）
            task_id: 任务 ID（loader 键，始终用原始 ID）
            target_dir: 目标根目录（会创建 input/ 和 output/ 子目录）
            use_symlink: 大数据湖 benchmark（Krama/LakeQA）用 symlink 代替整目录复制
            baseline_id: stage 到 Baseline 的目录名 / task.json.task_id（缺省用 task_id）。
                非 task_<int> 的 benchmark（FDA0001/legal-hard-1/lakeqa-full:EQA...）必须
                传 TaskIdMapper 映射后的 task_<int>，否则过不了 Baseline 的 _task_number 校验。
        """
        task = loader.load_task(task_id)
        src_context = task.context_dir
        staged_id = baseline_id or task_id
        task_input = target_dir / "input" / staged_id
        task_output = target_dir / "output" / staged_id

        task_input.mkdir(parents=True, exist_ok=True)
        task_output.mkdir(parents=True, exist_ok=True)

        # 准备 context：复制或符号链接
        dst_context = task_input / "context"
        if dst_context.exists():
            shutil.rmtree(dst_context)
        if use_symlink:
            try:
                dst_context.symlink_to(src_context, target_is_directory=True)
            except (OSError, NotImplementedError):
                # Windows 无管理员/开发者模式时 symlink 会失败 → 回退复制
                print(f"[warn] 无法创建 symlink（{dst_context}），回退为复制 context")
                shutil.copytree(src_context, dst_context)
        else:
            shutil.copytree(src_context, dst_context)

        # 写 task.json（必须恰好是 {task_id, difficulty, question}，Baseline 会校验）
        task_json = {
            "task_id": staged_id,
            "difficulty": task.difficulty,
            "question": task.question,
        }
        with (task_input / "task.json").open("w", encoding="utf-8") as f:
            json.dump(task_json, f, ensure_ascii=False, indent=2)

        # 写 gold.csv
        if task.gold_answer:
            with (task_output / "gold.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for row in task.gold_answer:
                    w.writerow(row)

        return task_input, task_output
