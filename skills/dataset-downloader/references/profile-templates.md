# Profile 模板与候选值

## 选择模板

| 需求 | Profile 模板 | Batch 模板 |
|---|---|---|
| ERA5 小时单层、近地面、云辐射、海浪、静态场 | [小时单层](../assets/templates/cds_hourly_single_profile.yaml) | [CDS Batch](../assets/templates/cds_batch.yaml) |
| ERA5 小时气压层 | [小时气压层](../assets/templates/cds_hourly_pressure_profile.yaml) | CDS Batch |
| ERA5-Land 小时 | [小时陆面](../assets/templates/cds_hourly_land_profile.yaml) | CDS Batch |
| ERA5 单层日统计 | [日统计单层](../assets/templates/cds_daily_single_profile.yaml) | CDS Batch |
| ERA5 气压层日统计 | [日统计气压层](../assets/templates/cds_daily_pressure_profile.yaml) | CDS Batch |
| ERA5-Land 日统计 | [日统计陆面](../assets/templates/cds_daily_land_profile.yaml) | CDS Batch |
| Marine ORAS5 日数据子集 | [ORAS5](../assets/templates/marine_oras5_profile.yaml) | [Marine Batch](../assets/templates/marine_batch.yaml) |

模板的注释列出当前仓库配置中的候选值，实际字段使用 `REPLACE_*` 占位符。
不要把全部候选值自动填入任务。将模板复制到用户任务目录，按需求选值；
完成后必须检查无占位符残留，再调用 Datakit 预览。

例如独立任务目录可以包含 `profiles/temperature.yaml` 和 `batch.yaml`；
Batch 的 `profiles[].path` 可写 `./profiles/temperature.yaml`。
日期和输出根目录放在 Batch 中，不写入 Profile。

## Batch 模板

[CDS Batch](../assets/templates/cds_batch.yaml) 适用于 CDS 小时和日统计 Profile；
[Marine Batch](../assets/templates/marine_batch.yaml) 适用于 Marine Profile。
两者不能混在同一个 Batch 中，调用入口时分别用 `--source cds` 或 `--source marine`。
模板不预设正式日期，所有 `REPLACE_*` 值均须填写。

| 配置 | 作用与填写规则 |
|---|---|
| `profiles[].path` | 引用实际 Profile；绝对路径或相对 Batch 文件的路径 |
| `profiles[].output_subdir` | 相对输出根目录的子目录；每个 Profile 使用不同子目录以避免冲突 |
| `storage.root_path` | 下载输出根目录。生成 Batch 前必须与用户确认；未指定时提议 `~/xmetai-downloads/<任务名>/` 的绝对路径，不使用 skill 仓库、`output/` 或源码目录。相对路径以 Batch 文件为基准 |
| `period.start_month/end_month` | CDS 首尾包含的月份，字符串 `YYYY-MM`；不支持只下载月内部分日期 |
| `period.start_date/end_date` | Marine 首尾包含的日期，字符串 `YYYY-MM-DD` |
| `execution.max_workers` | 大于等于 1 的整数；模板从单 worker 起步 |
| `execution.max_retries` | 首次尝试失败后的额外重试次数，非负整数 |
| `execution.retry_delay_seconds` | 失败后重试的等待秒数，非负数 |
| `execution.continue_on_error` | 任务失败后是否继续其他任务；不代表最终成功，不省略失败核对 |
| `records` | 可选的计划和状态记录路径；启用时同时提供 `plan_path`、`status_log_path` |

省略 `records` 时，记录放在
`storage.root_path/_records/<Batch 文件名不含扩展名>/plan.txt` 和 `status.jsonl`。
模板注释给出了自定义记录路径的写法；相对记录路径也以 Batch 文件为基准。
恢复下载时保留同一任务对应的状态记录和输出文件，不把仅有文件视为完成。

Batch 不配置 `lead_hour` 或 `temporal_resolution`。ERA5 小时时次由 Profile 的
`time` 选择；Marine 数据的原生时间分辨率由所选数据集决定。
`--dry-run`、`--plan-only`、`--execute` 属于入口命令参数，不写入 Batch。
模板只定义任务配置，不会自行启动下载；先预览并确认计划，再正式执行。

## 字段与语义

| 字段 | 适用范围 | 说明 |
|---|---|---|
| `dataset` / `dataset_id` | CDS / Marine | 外部数据集标识，不是 Datakit 标准产品名 |
| `variable` / `variables` | CDS / Marine | 非空、无重复的外部变量名列表；不能直接填 schema 短名 |
| `product_type` | ERA5 单层、气压层 | 当前示例为 `reanalysis`；ERA5-Land 不要求此字段 |
| `pressure_level` | CDS 气压层 | hPa 字符串列表；小时示例有 14 层，日统计示例有 13 层，见各模板 |
| `time` | CDS 小时 | 带引号的 `HH:00` 列表；示例为 00、06、12、18 UTC，不代表只能选四个时次 |
| `area` | CDS | 数值列表 `[北, 西, 南, 东]`；`null` 表示不裁剪，必须明确全球需求 |
| `statistics` | CDS 日统计 | 以统计方式分组，每组只含 `variable`；仅保留用户要求的组 |
| `frequency` | CDS 日统计 | 输入采样频率 `1_hourly` / `3_hourly` / `6_hourly`，不是输出频率 |
| `time_zone` | CDS 日统计 | 日界时区；示例为 `utc+00:00`，按用户需求选择 |
| `data_format` / `download_format` | CDS 小时 | 示例为 `netcdf` 配合 `zip` 或 `unarchived`；先核对对应数据集 |
| `longitude` / `latitude` | Marine | 分别包含数值 `minimum`、`maximum`，单位为度 |
| `depth` | Marine | 数值上下界，单位为米，不是气压层 |
| `file_tag` | Marine | 输出文件标签，由任务明确指定 |
| `dataset_version` / `dataset_part` / `service` | Marine 可选 | 查官方目录后才固定，不沿用示例历史值 |

日统计模板使用下载器支持的 `statistics` 分组结构。旧式顶层 `variable` +
`daily_statistic` 仍由下载器支持，但不能与 `statistics` 混用。
日统计不填写 `time`、`data_format` 或 `download_format`，输出为 NetCDF ZIP。
小时 `time` 的抽样不会自动把降水或辐射聚合成相同长度的累计量。

## 候选值的来源与边界

候选快照核对日期：**2026-10-27**。以下来源均相对用户指定的 Datakit 仓库根：

- 小时单层：`examples/dataset_downloader/cds_download/profiles/` 下的
  `reanalysis-era5-single-levels_{sfc,cldrad,wave,static}_h_v1.yaml`。
- 小时气压层：同目录 `reanalysis-era5-pressure-levels_pl_h_v1.yaml`。
- 小时陆面：同目录 `reanalysis-era5-land_soil_h6_v1.yaml`。
- 单层日统计：同目录 `derived-era5-single-levels-daily-statistics_` 开头的
  `sfc_mean`、`sfc_sum`、`t2m_max`、`t2m_min` 配置。
- 气压层日统计：同目录 `derived-era5-pressure-levels-daily-statistics_pl_mean.yaml`。
- 陆面日统计：仅在 `cds_download.py` 的 `DATASET_SPECS` 注册，
  **没有本地示例 Profile，因而不提供未经核对的变量候选值**。
- ORAS5：`examples/dataset_downloader/copernicusmarine_download/profiles/`
  `cmems_mod_glo_phy-all_my_0.25deg_P1D-m_oras5_daily_v1.yaml`。

这些候选值不是官方完整清单，不是自动同步的注册表，也不是变量与统计方式的
任意组合许可。海浪示例中仅在注释标记为 `pending_special_source` 的变量未纳入候选。
日最大/最小温的 `since_previous_post_processing` 变量不能替换成小时气温极值。

生成配置时以当前 Datakit 下载器规则核对字段，以数据集官方定义核对变量可用性，
以 schema 核对语义；有冲突就说明，不静默回退到示例。用户要求候选之外的变量时，
先检查数据集和下载器支持，不以“示例未列出”为由直接否定。
更新候选值时同步核对上述来源、模板注释与日期，不修改 Datakit 的示例配置。
