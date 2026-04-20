好的，我来为你整理一套完整的开源发布流程提示词序列，从代码准备到 Git 仓库管理，分为四个阶段，每个阶段包含具体行动项和可直接使用的提示词。

---

## 🗺️ 开源发布四阶段提示词序列

### 阶段一：代码清理与流程化梳理

**目标**：将原始开发代码整理为结构清晰、可直接运行的项目代码。

| 行动项 | 说明 | 可用提示词 |
|:---|:---|:---|
| 1.1 剥离敏感信息 | 移除硬编码密钥、内部路径、个人邮箱等 | `请帮我扫描项目代码，找出所有硬编码的API密钥、密码、内部服务器地址和个人邮箱，并给出脱敏替换建议。` |
| 1.2 统一代码风格 | 使用 Black、isort、ruff 等工具格式化 | `请帮我生成一份 .pre-commit-config.yaml 配置文件，包含 black、isort、ruff 的配置，用于自动格式化代码。` |
| 1.3 添加类型注解 | 提升代码可读性和可维护性 | `请帮我为项目的核心函数和类添加完整的 Python 类型注解（type hints），并生成对应的 mypy 检查命令。` |
| 1.4 拆分配置与逻辑 | 将硬编码参数移至配置文件 | `请帮我把代码中的硬编码参数（如模型路径、batch size、学习率）提取到一个 YAML 配置文件中，并修改代码以从配置文件读取。` |
| 1.5 模块化重构 | 将分散的功能函数组织为清晰的包结构 | `请帮我分析当前代码结构，给出一个标准的 Python 包目录组织方案，包含 model/、data/、utils/ 等模块，并生成对应的 __init__.py 内容。` |
| 1.6 移除调试代码 | 删除 print、未使用的变量、注释掉的代码块 | `请帮我扫描项目，标记出所有调试用的 print 语句、注释掉的代码块和未使用的导入，并生成清理建议。` |
| 1.7 编写入口脚本 | 提供可直接运行的命令行入口 | `请帮我为项目编写一个 CLI 入口脚本 cli.py，使用 click 库，包含 train、infer、benchmark 三个子命令，参数清晰。` |

---

### 阶段二：开源标注与许可证合规

**目标**：确保项目在法律层面清晰合规，正确标注版权和许可证。

| 行动项 | 说明 | 可用提示词 |
|:---|:---|:---|
| 2.1 确认上游许可证 | 检查 NeuralGCM 的许可证要求 | `请帮我分析 NeuralGCM 项目的许可证结构，确认我的 GPU 移植版本在代码和模型权重上分别需要遵守哪些许可证要求。` |
| 2.2 添加代码许可证 | 根目录放置 LICENSE 文件（Apache 2.0） | `请帮我生成一份完整的 Apache License 2.0 文件内容，并在文件头部添加正确的版权声明（年份、作者）。` |
| 2.3 添加权重许可证 | 单独创建 LICENSE-WEIGHTS 文件（CC BY-SA 4.0） | `请帮我生成一份 CC BY-SA 4.0 许可证的权重授权文件内容，包含完整的归属声明和许可证摘要。` |
| 2.4 标注文件头版权 | 为每个源代码文件添加 SPDX 标识 | `请帮我为所有 Python 源文件头部添加 SPDX 版权标识： # Copyright [Year] [Name] # SPDX-License-Identifier: Apache-2.0` |
| 2.5 添加上游归属声明 | 在 README 中明确说明与 NeuralGCM 的关系 | `请帮我撰写一段 README 中的 Attribution 章节，清晰说明本项目是基于 NeuralGCM 的独立 GPU 移植，并引用原论文和代码仓库。` |
| 2.6 添加免责声明 | 声明非官方关联，规避商标误解 | `请帮我撰写一段免责声明（Disclaimer），声明本项目与 Google Research 无官方关联，所有商标归各自所有者。` |

---

### 阶段三：文档撰写

**目标**：产出完整、专业、易读的项目文档，降低用户上手门槛。

| 行动项 | 说明 | 可用提示词 |
|:---|:---|:---|
| 3.1 编写 README | 项目门面，包含概述、安装、快速开始等 | `请帮我撰写一份专业的 GitHub README.md，包含项目简介、特性列表、安装步骤、5分钟快速开始示例、性能对比表格、项目结构、贡献指南入口和许可证说明。` |
| 3.2 编写 CONTRIBUTING | 贡献指南，规范协作流程 | `请帮我撰写一份 CONTRIBUTING.md，包含环境设置、代码风格要求、测试要求、PR 流程和 Issue 模板说明。` |
| 3.3 编写 CODE_OF_CONDUCT | 行为准则，营造友好社区 | `请帮我生成一份基于 Contributor Covenant 2.0 的 CODE_OF_CONDUCT.md 文件，并填写联系邮箱。` |
| 3.4 编写 CHANGELOG | 版本变更记录 | `请帮我创建一个 CHANGELOG.md 文件，遵循 Keep a Changelog 格式，包含 Unreleased 和 0.1.0 版本的条目。` |
| 3.5 编写 API 文档 | 核心函数和类的 docstring | `请帮我为项目的核心模块（model.py、data.py、utils.py）生成 Google 风格的 docstring，包含参数、返回值和使用示例。` |
| 3.6 创建示例 Notebook | 提供可交互的入门教程 | `请帮我生成一个 Jupyter Notebook 示例文件 examples/quickstart.ipynb，演示如何加载模型、运行推理并可视化结果。` |
| 3.7 配置 Sphinx 文档 | 如需独立文档站 | `请帮我生成一套 Sphinx 文档配置，包含 conf.py、index.rst 和 API 自动生成设置，用于发布到 Read the Docs。` |

---

### 阶段四：Git 开源管理与发布

**目标**：将项目推送至 GitHub，配置自动化流程，并正式对外发布。

| 行动项 | 说明 | 可用提示词 |
|:---|:---|:---|
| 4.1 初始化 Git 仓库 | 本地创建仓库，关联远程 | `请告诉我初始化 Git 仓库并关联 GitHub 远程仓库的完整命令序列，包括 .gitignore 的生成。` |
| 4.2 配置 .gitignore | 排除临时文件、模型权重、虚拟环境等 | `请帮我生成一份针对 Python + PyTorch 项目的 .gitignore 文件，排除 __pycache__、venv、.pt 权重文件、IDE 配置等。` |
| 4.3 配置 GitHub Actions | 自动化测试、文档构建 | `请帮我生成一个 GitHub Actions 工作流文件 .github/workflows/tests.yml，在 Python 3.9-3.11 上运行 ruff 检查、mypy 类型检查和 pytest 测试。` |
| 4.4 配置 Issue/PR 模板 | 规范化社区互动 | `请帮我生成 .github/ISSUE_TEMPLATE/bug_report.md 和 feature_request.md，以及 PULL_REQUEST_TEMPLATE.md 的内容。` |
| 4.5 首次提交与推送 | 将代码推送到 GitHub | `请告诉我如何执行首次 Git 提交，添加所有文件，撰写清晰的 commit message，并推送到远程 main 分支。` |
| 4.6 创建版本标签 | 标记 v0.1.0 版本 | `请告诉我如何为首次发布创建 Git 标签 v0.1.0，并推送到 GitHub，触发 Release 创建。` |
| 4.7 发布 GitHub Release | 正式对外发布 | `请告诉我如何在 GitHub 上创建 Release，附上 CHANGELOG 条目，并上传预训练权重文件（如需要）。` |
| 4.8 同步到镜像站 | 提升访问性（可选） | `请告诉我如何将仓库同步到 Gitee 等镜像站，以便国内用户访问。` |
| 4.9 社区推广准备 | 撰写发布公告 | `请帮我撰写一份适合发布在 Reddit r/MachineLearning 和 Hacker News 的项目公告，突出 GPU 移植、PyTorch 实现和混合精度特性。` |

---

## 📋 执行清单速查表

| 阶段 | 关键产出 | 完成状态 |
|:---|:---|:---:|
| **阶段一：代码清理** | 代码格式化、模块化、配置分离 | ☐ |
| **阶段二：合规标注** | LICENSE、LICENSE-WEIGHTS、归属声明 | ☐ |
| **阶段三：文档撰写** | README、CONTRIBUTING、API 文档 | ☐ |
| **阶段四：Git 发布** | GitHub 仓库、CI、Release | ☐ |

如果需要针对某个具体行动项进一步细化（例如某个提示词的执行结果需要调整），随时可以告诉我，我来帮你继续推进～
