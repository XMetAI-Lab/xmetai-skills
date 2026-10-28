# CDS 下载

以下路径相对 Datakit 仓库根，而非 skill 的安装目录：

- `examples/dataset_downloader/cds_download/README_download.md`：请求规则和输出约定。
- `examples/dataset_downloader/cds_download/profiles/`：变量组合示例，不是完整可用清单。
- `examples/dataset_downloader/cds_download/cds_download.py`：`DATASET_SPECS` 为数据集支持规则。
- `examples/dataset_downloader/cds_download/requirements.txt`：唯一依赖清单。

需要时在仓库根安装：

```bash
python -m pip install -r examples/dataset_downloader/cds_download/requirements.txt
```

## 来源与认证

来源是 CDS（https://cds.climate.copernicus.eu/）。脚本调用 `cdsapi.Client().retrieve()`，
服务地址和认证由官方客户端的环境配置或用户 `.cdsapirc` 管理，不是 Profile 字段。
提供以下三种选项，不让用户通过聊天提交 token。

### 选项一：本地交互向导

由用户在实际运行 Datakit 的机器和用户账号下，打开私有终端执行：

```bash
python <skill>/scripts/setup_cds.py
```

选择 `1`，按提示打开官方个人资料页获取 Personal Access Token，在终端隐藏输入。
向导使用官方 CDS API 地址，自动识别系统并通过 `Path.home()` 定位配置文件：
Linux/macOS 为 `~/.cdsapirc`，Windows 为 `%USERPROFILE%\.cdsapirc`，不是项目目录
或 Windows AppData。POSIX 权限为 0600；Windows 通过系统 PowerShell 设置并检查
仅当前用户可访问的 ACL，先保护空临时文件再写 token，失败就停止。
脚本以此权限原子保存配置，
不打印 token、不读取已有文件内容、不保存旧凭据备份。已有普通文件须确认
**整文件替换**，拒绝符号链接和非普通文件；取消或输入不合法时不保存。
支持 Windows 和 Linux/macOS 等 POSIX 环境，不具备隐藏输入条件就停止。
Windows 需要可用的 `powershell.exe` 和支持 ACL、硬链接的本地文件系统；
不支持时明确报错，不降级为不受保护的保存。
配置只作用于执行脚本的机器和用户；WSL、远程 SSH、容器中的主目录与宿主机不同。

这条命令必须由用户自己在私有终端运行，不在 AI 的 Bash 工具、捕获输出的任务、
共享终端录制或管道中处理密钥，也不通过自动输入工具代填。
conda 环境可先 `conda activate xmetai-skills` 再执行，避免捕获输出的包装方式。
工具仅使用标准库，不依赖 `cdsapi` 或 PyYAML。

若设置了 `CDSAPI_RC`、`CDSAPI_URL` 或 `CDSAPI_KEY`，它们可能影响客户端配置；
向导会提示变量名并拒绝写默认文件。用户私下检查这些覆盖，或改用手动路径。
不要通过打印环境变量或配置文件来排错。

### 选项二：手动配置指导

```bash
python <skill>/scripts/setup_cds.py --manual
```

也可在交互菜单选择 `2`。只输出说明，不读写凭据、不联网。
用户按 [CDS API 官方说明](https://cds.climate.copernicus.eu/how-to-api)，
在私有本地编辑器中填写用户级配置；Linux/macOS 执行 `chmod 600 ~/.cdsapirc`，
Windows 在文件“属性 → 安全 → 高级”中禁用继承并仅保留当前用户的访问权限。
Windows 文件名须为 `.cdsapirc`，不要误存为 `.cdsapirc.txt`。

### 选项三：全手动（零代码执行）

用户选择此项后，skill 直接在对话中给出以下步骤，**认证配置阶段不执行任何代码**：
不运行 `setup_cds.py`（包括 `--manual`）、Shell 命令、依赖安装、环境探测或在线验证，
不读取或修改凭据文件。无需安装 Python 或运行向导即可阅读并完成这些步骤。

1. 在实际下载机器、实际用户账号下配置。若使用 WSL、SSH 或容器，配置的是该环境，
   而非本机 Windows 用户目录；不确定环境时先澄清。
2. 用户自行在浏览器登录 [CDS 个人资料页](https://cds.climate.copernicus.eu/profile)，
   获取 Personal Access Token；不将 token 发送到聊天。
3. 用私有文本编辑器创建或编辑用户主目录中的 `.cdsapirc`：
   - **Linux/macOS**：`~/.cdsapirc`，例如 `/home/<用户名>/.cdsapirc`。
   - **Windows**：`%USERPROFILE%\.cdsapirc`，例如
     `C:\Users\<用户名>\.cdsapirc`。保存时选择“所有文件”，不要追加 `.txt`。
   文件格式如下，用户只在自己的编辑器中替换占位符：

   ```yaml
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR_PERSONAL_ACCESS_TOKEN>
   ```

4. 由用户自行限制文件访问：
   - **Linux/macOS**：只允许所有者读写（0600），可在文件管理器权限设置中完成；
     若用户选择终端方式，可自行执行 `chmod 600 ~/.cdsapirc`，skill 不代执行。
   - **Windows**：文件“属性 → 安全 → 高级”中禁用权限继承，仅保留当前用户访问。
   不在共享编辑会话、截图、终端录制或版本库中暴露配置。
5. 用户自行在目标数据集页面接受使用条款；如已有 `CDSAPI_RC` / `CDSAPI_URL` /
   `CDSAPI_KEY` 覆盖，私下确认与所用配置一致，不打印其值。
6. 用户只需回复“已配置完成”。skill 等待该确认后，从原来的 Profile/Batch、
   预览或执行流程断点继续，不要求额外跑向导、展示凭据或提交配置证明。

“已配置完成”只是用户确认，不等于服务端认证验证通过。配置完成不自动授权下载，
后续代码执行仍遵循用户原有授权；首次正式下载仍须确认具体计划。

三种选项均不代替账号注册、浏览器登录、获取 token 和接受每个目标数据集的条款。
向导写入的 `url` / `key` 结构依据
[官方 cdsapi 配置说明](https://github.com/ecmwf/cdsapi#configure)。
**保存成功不代表认证有效**，离线预览也不验证凭据；在线验证必须单独确认请求范围，
不得为测试认证自动运行整个 Batch。

## 排队与执行终端

正式提交前按 [Skill 的终端选择流程](../SKILL.md#cds-排队提示与执行终端选择)
在正文展示“下载总览”和简短排队提示，再用简短问题确认下一步；
不要以“预览通过”开场或把参数、长路径、排队术语堆入选择框。
提供 Agent 执行、其他终端手动执行及暂不下载的选择；选择针对本次总览中的具体计划，
不增加下载器参数，也不自动授权额外下载或认证探测。

- 提交前只需提示“提交后可能需要等待 CDS 排队处理；文件大小以实际下载结果为准”，
  不提前解释尚未出现的服务端状态，不编造体积或完成时间。
- 实际收到 `accepted` 时再说明“CDS 已接收请求，正在等待处理”，不等于正在提取数据
  或已开始传输；日志可能只在状态变化时更新。小请求也可能排队，不能承诺换日期或
  数据集就更快。
- [官方 Live Dashboard](https://cds.climate.copernicus.eu/live) 的全站统计不等于
  本请求名次或等待时间。未核实的数据不报告为实时值；无法取得队列数量、排位或
  预计时间时明确说明不可用，不从请求 ID、文件大小或全站排队量推算。
- Agent 终端等待期间保留实际请求 ID、最后观测状态和时间依据。没有完成通知或
  最终汇总时不宣称成功；不因超过两分钟自动停止或重提请求。
- 其他终端模式只交付实际可复制的命令，使用同一环境、用户与已预览 Batch，
  不含凭据，不默认 `--force-redownload`；AI 不同时执行同一任务。
- 已提交任务不能靠重跑 Batch 无缝接管。停止本地等待不保证服务端取消；
  切换终端前须核对旧任务，避免重复提交。任务结束后仍须核对记录并生成报告。

## 请求和文件

从 [Profile 模板](./profile-templates.md) 生成任务配置，不默认复用上述示例。
气压层、变量、时次等候选见模板注释；字段按当前下载器核对，变量按官方数据集核对。

- 小时任务按“变量 × 月”拆分，指定时次和所有所选气压层在同一请求里。
- `period.start_month/end_month` 首尾包含；不会自动寻找最新月份。
- `area: null` 不裁剪区域；不能把它解释成中国或其他局地。
- 日统计由服务端计算，按“变量 × 统计方式 × 月”请求，返回 NetCDF ZIP。
- 日统计的 `frequency` 是输入采样频率，不是输出频率。
- 最高 / 最低温的 since_previous_post_processing 变量不等同于对小时 t2m 直接取极值。
- 数据文件路径、临时文件与记录布局以当前下载器 README 和实际预览为准。

不要直接使用尚未检查的默认 Batch：当前 `batch_daily_task.yaml` 引用的合并
Profile 尚未齐备。按本次需求从技能模板生成 Profile 和 Batch，检查实际引用文件。
不因为文档出现某个文件名就假设该文件存在。
