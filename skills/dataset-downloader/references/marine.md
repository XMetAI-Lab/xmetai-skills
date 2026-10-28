# Copernicus Marine 下载

以下路径相对 Datakit 仓库根：

- `examples/dataset_downloader/copernicusmarine_download/README.md`：使用和输出约定。
- `examples/dataset_downloader/copernicusmarine_download/profiles/`：数据集与变量示例。
- `examples/dataset_downloader/copernicusmarine_download/copernicusmarine_download.py`：执行实现。
- `examples/dataset_downloader/copernicusmarine_download/requirements.txt`：唯一依赖清单。

需要时在仓库根安装：

```bash
python -m pip install -r examples/dataset_downloader/copernicusmarine_download/requirements.txt
```

## 来源与认证

来源是 Copernicus Marine（https://data.marine.copernicus.eu/）。
官方 `copernicusmarine` 库根据 `dataset_id` 查询目录并确定访问服务，
本脚本不维护文件下载 URL。当前执行器调用 `login()`，之后调用 `subset()`。
默认允许用户在下载器调用 `login()` 时交互输入用户名、密码并确认保存。
正式下载命令应由用户在实际下载机器和用户账号下的可交互私有终端执行；
AI 负责准备命令与配置，不通过捕获输出的工具、自动输入工具或聊天处理密码。
也可提前执行 `copernicusmarine login`，但当前执行器仍会再次调用 `login()`，
是否复用或提示覆盖由官方客户端决定，不保证无人值守。
凭据由官方工具管理，与 CDS 分开，不放入聊天、源码、Profile、Batch 或报告。

用户确认运行结束后，skill 继续核对任务计划、匹配的状态记录及目标文件，并按
[下载报告模板](../assets/templates/download_report.md) 在实际任务记录目录保存报告。
无需用户发送登录截图或凭据。不能取得本次退出码或汇总时如实标注未知，
仅凭“已结束”或历史完成记录不宣称本次全部成功。数据内容未检查时明确注明。

## 请求和文件

从 [ORAS5 Profile 模板](../assets/templates/marine_oras5_profile.yaml) 生成任务配置。
注释包含当前示例的八个变量及经纬度、深度范围，仅作候选说明，不默认全部采用。

- 当前随附 Profile 是 ORAS5；变量、范围和深度以实际 Profile 为准。
- `period.start_date/end_date` 是首尾包含的日历日期。
- 一天一次任务，所选变量写入当天一个 NetCDF，不是每个变量一个文件。
- 地理范围使用 `longitude` / `latitude` 的 `minimum`、`maximum`；
  深度使用 `depth` 的 `minimum`、`maximum`，不能套用 CDS 的 `area` / `pressure_level`。
- 可选 `dataset_version`、`dataset_part`、`service`；未固定时由官方工具选择。
  需要可复现版本时先核对官方目录，不能编造版本或服务名称。
- 临时输出以 `.part.nc` 结尾，成功后替换最终文件。

离线预览仅验证本地任务构建，不证明远端当前仍提供该版本或日期的数据。
