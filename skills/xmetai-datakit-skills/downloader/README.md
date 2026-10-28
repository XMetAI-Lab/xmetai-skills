# XMetAI Downloader

让 AI 按 Datakit 的变量约定，把下载需求变成配置、计划和本地原始文件。
能力契约和执行规则的唯一权威是 [SKILL.md](./SKILL.md)。

## 定位与目录结构

本 skill 属于 Datakit 技能族，**依赖 Datakit 仓库**，调用仓库内已有下载器，
不复制下载实现，不需要单独的 pyproject 或重复依赖清单。

```text
downloader/
├── SKILL.md
├── README.md
├── agents/
│   ├── openai.yaml
│   └── claude.yaml
├── assets/templates/
│   ├── cds_batch.yaml
│   ├── cds_hourly_single_profile.yaml
│   ├── cds_hourly_pressure_profile.yaml
│   ├── cds_hourly_land_profile.yaml
│   ├── cds_daily_single_profile.yaml
│   ├── cds_daily_pressure_profile.yaml
│   ├── cds_daily_land_profile.yaml
│   ├── marine_oras5_profile.yaml
│   ├── download_report.md
│   └── marine_batch.yaml
├── references/
│   ├── cds.md
│   ├── marine.md
│   ├── profile-templates.md
│   └── variables.md
└── scripts/
    ├── setup_cds.py
    ├── run_download.py
    └── render_download_report.py
```

## 使用

用户可以说：“下载指定区域 2020 年的 ERA5 2 米气温，每天四个时次，先给我看计划。”
Skill 会澄清需求、检查变量约定、从自带模板生成任务专用 Profile 和 Batch，
再调用 Datakit 的下载器。Datakit examples 中的 Profile 仅供参考，不作为默认配置。

模板包含 `variable` / `variables`、`pressure_level`、`time`、区域、日统计、
深度和格式等适用字段，并在注释列出当前 Datakit 配置的候选值及来源日期。
候选不是官方全集，不自动全部启用；详见
[Profile 模板与候选值](./references/profile-templates.md)。

生成 Batch 前先与用户确认 `storage.root_path`；未指定时提议
`~/xmetai-downloads/<任务名>/`，不要默认写到 skill 仓库、`output/` 或 Datakit 源码目录。

先将选中的 Profile 模板和 Batch 模板复制到任务目录，按需求替换占位符，
让 Batch 引用新 Profile。在 xmetai-skills 仓库根执行：

```bash
python skills/xmetai-datakit-skills/downloader/scripts/run_download.py \
  --repo-root /absolute/path/to/xmetai-datakit \
  --source cds /absolute/path/to/batch.yaml
```

默认离线预览；加 `--execute` 才正式下载。Marine 使用 `--source marine`。
完整参数用 `--help` 查看。把 skill 复制或注册到其他位置后，入口仍需访问
Datakit，可用 `--repo-root /absolute/path/to/xmetai-datakit` 显式指定。

入口只使用 Python 标准库；底层依赖分别由 Datakit 仓库内的
`examples/dataset_downloader/cds_download/requirements.txt` 和
`examples/dataset_downloader/copernicusmarine_download/requirements.txt` 维护。
默认不会安装依赖、登录或发起下载。认证准备见成员参考文档。

## CDS 排队与执行终端

正式提交前，skill 会提示 CDS 可能排队，并询问执行方式：

- **Agent 终端执行（默认）**：计划确认和认证准备完成后，由 AI 执行；
  等待期间按实际日志说明请求 ID、最后观测状态和已等待时长。
- **其他终端手动执行**：提供已预览 Batch 的正式命令及所需环境，由用户在同一
  下载机器和账号下运行，AI 不同时提交。命令使用实际绝对路径，不含凭据。
  用户报告结束后由 skill 核对记录和数据并生成报告。

`accepted` 只表示请求已接受，小区域、小文件也可能等待；日志不更新不一定是卡死。
[CDS Live Dashboard](https://cds.climate.copernicus.eu/live) 展示的全站排队量
不等于本请求排位，无法取得的队列数量或预计时间会标为不可用，不承诺哪个年月、
single levels 或 pressure levels 更快。不会因排队超过两分钟自动停止或换终端。

已运行任务切换终端前必须核对旧请求：停止本地进程不保证取消服务端任务，重跑
Batch 可能重复提交，不能在两个终端并行执行同一 Batch。
具体交互见 [Skill 流程](./SKILL.md#cds-排队提示与执行终端选择) 和
[CDS 说明](./references/cds.md#排队与执行终端)。
此默认值不适用于凭据输入或 Marine 的私有交互认证。

## CDS API key 配置

提供三种选项，无需阅读下载器代码：

- **本地交互向导**：用户在实际下载机器的私有终端运行
  `python skills/xmetai-datakit-skills/downloader/scripts/setup_cds.py`，选择 `1`。
  自动识别系统，隐藏输入 token，保存到官方客户端默认读取的用户主目录：
  Linux/macOS 为 `~/.cdsapirc`（0600），Windows 为 `%USERPROFILE%\.cdsapirc`
  （仅当前用户可访问的 ACL，使用 Windows PowerShell 设置并检查）；
  已有配置须确认整文件替换。不在聊天、AI 工具或捕获输出的任务中输入密钥。
- **手动配置指导**：同一命令加 `--manual`，或菜单选择 `2`，只显示配置步骤。
- **全手动（零代码执行）**：skill 直接在对话中说明 Windows/Linux 的配置文件位置、
  `url` / `key` 格式、文件权限和数据集条款。认证配置阶段不执行任何代码，
  不运行向导（也不运行 `--manual`）、不检查或读写凭据文件。
  用户自行配置并回复“已配置完成”后继续原任务，不需要发送 token、截图或文件内容。
  用户确认不等于认证验证通过，也不自动授权下载。

全手动方式的完整步骤见 [全手动配置](./references/cds.md#选项三全手动零代码执行)；
脚本仍保留原来的两个菜单选项，全手动方式不需要进入脚本菜单。

Windows 在写入 token 前先保护空临时文件，缺少 PowerShell 或权限设置失败就停止。
脚本保存的是用户获取的 token，不生成 API key；工具不联网、不验证 key、不安装依赖。
WSL、SSH 或容器中运行时，配置保存到该运行环境的用户主目录，而非宿主 Windows 主目录。
存在客户端配置环境变量覆盖时会停止并提示，不打印其值。
账号登录、获取令牌及接受数据集条款仍由用户完成。
完整行为与边界见 [CDS 认证说明](./references/cds.md#来源与认证)。

## Marine 交互与下载报告

Marine 正式执行会调用官方 `copernicusmarine.login()`。用户可直接在此时输入账号密码，
不必先独立运行登录命令；为保护认证交互，skill 将正式命令交给用户在可交互的私有终端
运行，不通过 AI 捕获输出或代填密码。运行结束后继续由 skill 核对和交付。

CDS 与 Marine 正式执行结束后，skill 按
[下载报告模板](./assets/templates/download_report.md) 生成 Markdown 报告，保存到实际
计划文件所在的任务记录目录。默认位置为
`<storage.root_path>/_records/<Batch名>/download_report_<UTC时间戳>.md`，
使用带微秒的时间戳保留每次报告，不覆盖历史报告。

执行后可显式运行
[scripts/render_download_report.py](./scripts/render_download_report.py) 生成报告初稿：

```bash
python skills/xmetai-datakit-skills/downloader/scripts/render_download_report.py \
  /absolute/path/to/batch.yaml \
  --run-log /absolute/path/to/run.log --exit-code 0 \
  --request-id <request-id> --datakit-root /absolute/path/to/xmetai-datakit \
  --source cds --strict
```

脚本读取 Batch/Profile、计划、`status.jsonl`、可选运行日志和文件属性，不联网、
不读取凭据、不验证文件内容。它只生成初稿；AI 必须核对证据并补充未自动检查项。
未提供退出码或运行汇总时，报告会明确标注未知，不用 0 或成功兜底。
脚本使用 PyYAML；底层下载器预览已需要该依赖。

报告包括下载内容、请求参数、配置与数据绝对路径、完成/跳过/失败统计、
实际目标文件与字节大小、异常和恢复建议。多个 Profile 分别描述；文件多时另附完整清单。
失败或中断也生成如实记录的报告，不仅报告成功任务。
无法获取的退出码、本次统计、版本或元数据标为未知，不根据历史记录编造本次结果。
未实际检查数据内容时，明确注明未做科学完整性验收。

这是 **AI 按 skill 工作流生成并核对的报告**；`run_download.py` 不会自动生成 Markdown
报告，报告初稿由 `render_download_report.py` 显式生成。报告生成不涉及读取凭据，
也不会自动启动后处理或额外下载。完整规则以 [SKILL.md](./SKILL.md#6-核对与交付) 为准。

## 测试

测试统一保存在 xmetai-skills 仓库根目录的
[tests/test_downloader_skill.py](../../../tests/test_downloader_skill.py)。
在 xmetai-skills 仓库根目录执行：

```bash
python -m pytest -q tests/test_downloader_skill.py
```

仅测试 `scripts/run_download.py` 的参数门控、路径定位、命令转发与退出码传递。
使用临时桩程序替代实际下载器，不检查文档或模板，不调用真实 CDS / Marine 实现。
只需要 pytest，不需要 Datakit 仓库、认证或网络。

报告渲染器测试位于
[tests/test_downloader_report.py](../../../tests/test_downloader_report.py)，
使用临时 Batch、计划、状态和文件证据，不联网、不读取凭据：

```bash
python -m pytest -q tests/test_downloader_report.py
```

本地配置向导测试使用临时用户目录和假 token，不访问真实凭据：

```bash
python -m pytest -q tests/test_downloader_cds_setup.py
```

覆盖交互/手动路径、隐藏输入、覆盖确认、权限、取消、环境变量覆盖和写入失败。

模板测试位于 [tests/test_downloader_templates.py](../../../tests/test_downloader_templates.py)，
使用 PyYAML 检查七类模板替换占位符后的字段和类型；未安装 PyYAML 时跳过该文件。
可指定 Datakit 仓库，额外使用真实任务构建器和入口 `--dry-run` 检查请求、目标和计划：

```bash
XMETAI_DATAKIT_ROOT=/absolute/path/to/xmetai-datakit \
  python -m pytest -q tests/test_downloader_skill.py tests/test_downloader_templates.py
```

未指定 `XMETAI_DATAKIT_ROOT` 时仅跳过真实 Datakit 兼容性测试。
不登录、不连接远端；这不是官方变量可用性或文件内容验收。
ERA5-Land 日统计只检查本地接口结构，其变量仍需单独核对官方数据集。

### 可选 CDS 在线 smoke test

配置统一放在 [tests/fixtures/downloader/](../../../tests/fixtures/downloader/)，
测试位于 [tests/test_downloader_cds_live.py](../../../tests/test_downloader_cds_live.py)。
默认只检查配置，在线用例跳过；即使机器上已有 key，也不会自动联网。

用户先配置 CDS 凭据并接受 ERA5 数据集条款。在运行测试的环境中安装 Datakit
CDS requirements 以及文件检查依赖 `xarray`、`numpy`、`h5netcdf`、`h5py`、`pytest`。
在 xmetai-skills 仓库根目录显式授权在线测试：

```bash
XMETAI_RUN_CDS_LIVE=1 \
XMETAI_DATAKIT_ROOT=/absolute/path/to/xmetai-datakit \
CDSAPI_RC=/absolute/path/to/.cdsapirc \
  conda run -n xmetai-skills python -m pytest -q tests/test_downloader_cds_live.py
```

`CDSAPI_RC` 可省略，此时使用用户主目录的 `.cdsapirc`。仅提供凭据路径，
不要把 key 放入命令、测试配置或版本库；测试不读取或展示凭据内容。
显式启用后，缺少配置或依赖、认证失败、未接受条款、服务端错误都会使测试失败，
不会以跳过代替失败。

每次在线用例提交一个真实请求：2021 年 2 月、每天 00 UTC 的 2 米气温，
北纬 30–30.25°、东经 120–120.25°，无重试，子进程最长等待 600 秒。
超时只会终止本地等待，不保证取消已经提交的服务端任务。
测试在 pytest 临时目录写入独立 Batch 和下载结果，不覆盖原 smoke 下载，
并检查完成记录、计划、28 个时刻、2×2 网格、K 单位和 112 个有限值。
离线配置文件也可直接配合入口预览；该测试 fixture 的默认输出为仓库
`output/downloader_smoke`，仅用于本地 smoke，不作为 skill 的默认下载目录。

## 注册

仅把目录放进仓库，不保证宿主会自动发现。可在 xmetai-skills 仓库根执行：

```bash
mkdir -p .github/skills
ln -s ../../skills/xmetai-datakit-skills/downloader .github/skills/downloader
```

已有同名注册时先检查，不覆盖。其他宿主按其技能发现规则注册本成员目录。
注册不等于安装运行依赖。
