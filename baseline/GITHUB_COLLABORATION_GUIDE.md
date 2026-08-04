# GitHub 协作指南

共享仓库：`https://github.com/Mingshu-Ao/data-agent-evaluation`

推荐目录：

```text
data-agent-evaluation/
  baseline/
  pipeline/
  contracts/
  shared_suites/
```

不要向当前 `HKUSTDial/kddcup2026-data-agents-starter-kit` 官方 origin 推送个人修改。

## 首次上传 Baseline

```powershell
$git = "C:\Users\15120\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe"
& $git --version

cd D:\bupt\codex_project\data_agent
& $git clone https://github.com/Mingshu-Ao/data-agent-evaluation.git
cd .\data-agent-evaluation

New-Item -ItemType Directory -Path .\baseline
Expand-Archive `
  D:\bupt\codex_project\data_agent\handoff\baseline_pipeline_integration_2026-08-04.zip `
  .\baseline

& $git add baseline
& $git commit -m "feat: add baseline integration package"
& $git push -u origin main
```

若仓库已有 README，先执行 `& $git pull origin main` 再提交。

## Pipeline 同学

```powershell
$git = "C:\Users\15120\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe"
& $git clone https://github.com/Mingshu-Ao/data-agent-evaluation.git
cd .\data-agent-evaluation
& $git switch -c feature/pipeline-integration

# 将代码放入 pipeline/，共同 schema 放入 contracts/。
& $git add pipeline contracts shared_suites
& $git commit -m "feat: add unified benchmark pipeline"
& $git push -u origin feature/pipeline-integration
```

在 GitHub 创建 Pull Request 后再合并到 `main`。数据集、视频、API Key、SSH/VPN 配置和完整 artifacts 不上传。
