# FICC 看板 · 换电脑恢复手册

> 换公司/换电脑时照此操作即可恢复网站访问与每日自动更新。
> 仓库地址：https://github.com/Ericshunjian/ficc-dashboard

---

## 一、访问网站（零成本，最先做）

直接浏览器打开：**https://ericshunjian.github.io/ficc-dashboard/**

- 历史数据（2021-06-03 至最近一次更新）全部可看，无需任何配置。
- 仅"访问"到此结束；若要恢复"每日自动更新"，继续下面步骤。

---

## 二、Clone 仓库

```bash
git clone git@github.com:Ericshunjian/ficc-dashboard.git
```

仓库含：7 个 HTML 页面 + 4 个 JSON 数据 + 全部 Python 脚本 + `data_source/`（历史数据源 xlsx）。

---

## 三、安装 Python + pandas

1. 安装 Python 3.13+（https://www.python.org/downloads/）
2. 安装依赖：
```bash
pip install pandas openpyxl
```

---

## 四、数据源准备（关键）

每日更新依赖 4 个 xlsx 源文件。仓库 `data_source/` 目录已附带**历史快照**，用于初始化：

| 仓库内文件（data_source/） | 用途 | 需放到的新路径 |
|---|---|---|
| `bond_data.xlsx` | 机构行为数据（2026-） | `<新路径>/bond_data.xlsx` |
| `bond_data_备份截至2025年.xlsx` | 机构行为历史（2021-2025） | `<新路径>/bond_data_备份截至2025年.xlsx` |
| `FICC原始数据（现券）.xlsx` | 个券收益率 | `<新路径>/FICC原始数据（现券）.xlsx` |
| `FICC原始数据（衍生品、收益率曲线）.xlsx` | 收益率曲线 | `<新路径>/FICC原始数据（衍生品、收益率曲线）.xlsx` |

**两种方式任选其一：**

- **方式 A（推荐）**：把 `data_source/` 的 4 个文件复制到你新电脑习惯的目录，然后修改 `daily_update.py` 顶部 line 28-41 的路径常量指向新目录。
- **方式 B**：直接修改 `daily_update.py` 顶部路径常量，指向 `data_source/` 目录（相对仓库根目录），省去复制。但注意 `data_source/` 是历史快照，需每天用新下载的文件覆盖才能更新。

### daily_update.py 需要修改的路径常量（line 28-41）

```python
BOND_DATA_EXCEL       # 机构行为数据（当天）
OLD_BOND_DATA_EXCEL   # 机构行为历史备份
FICC_EXCEL            # FICC 现券收益率
CURVE_EXCEL           # 收益率曲线
USER_PREPROCESS_SCRIPT  # 用户预处理脚本路径（若不跑预处理可置空跳过）
USER_PREPROCESS_CWD     # 预处理脚本工作目录
```

> 预处理脚本 `现券数据处理_2026.py` 负责从原始日报生成 `bond_data.xlsx`。若新环境下没有原始日报，可跳过预处理（daily_update 会直接用现成的 bond_data.xlsx），把 `USER_PREPROCESS_SCRIPT` 置空即可。

---

## 五、SSH Key 配置（push 到 GitHub 用）

```bash
# 1. 生成 key
ssh-keygen -t ed25519 -C "your_email@example.com"
# 一路回车（默认路径 ~/.ssh/id_ed25519，可不设密码）

# 2. 查看公钥，复制输出
cat ~/.ssh/id_ed25519.pub

# 3. 添加到 GitHub
#    GitHub → Settings → SSH and GPG keys → New SSH key → 粘贴公钥

# 4. 测试
ssh -T git@github.com   # 应显示 "Hi Ericshunjian!"
```

---

## 六、运行每日更新

```bash
cd ficc-dashboard
python daily_update.py
```

脚本会自动：[0]预处理(可选) → [1]机构行为 → [2]FICC现券 → [3]曲线 → [4]合并 → [5]因子 → 自动 git push 到 github。

成功后 GitHub Pages 约 1-2 分钟生效，网站数据更新。

> 若自动 push 失败（SSH/网络），手动：
> ```bash
> git add -A && git commit -m "data: 每日更新" && git push github main
> ```

---

## 七、研究结论库云同步配置（防丢）

研究结论库（`research_conclusions.html`）默认存浏览器 localStorage，换电脑/换浏览器会丢。已支持云端同步：

1. 打开网站 → 进入"研究结论库"页 → 点右上角 **☁ 云同步**
2. 填写配置：
   - Owner / Repo / Branch / 文件路径（默认已填好：Ericshunjian / ficc-dashboard / main / research_conclusions.json）
   - **Personal Access Token**：GitHub → Settings → Developer settings → Personal access tokens → Generate new token（勾选 **repo** 权限）→ 粘贴
3. 点 **⬆ 上传到云**：把本地结论推到仓库 `research_conclusions.json`
4. 保存配置

**换电脑后**：打开研究结论库页，会**自动从云端拉取**最新结论合并到本地（无需 token，公开仓库 raw 读取）。修改后点"上传到云"同步回去。

> Token 仅存本机 localStorage，不会上传。换电脑需重新填 token（下载不需要 token，只有上传需要）。

---

## 八、日常更新流程（新电脑上稳定运行后）

每天 9:00 后（数据源就绪）：

```bash
cd ficc-dashboard
python daily_update.py
```

或直接对 WorkBuddy 说"更新 FICC"（若使用 WorkBuddy 助手）。

---

## 九、关键路径速查

| 用途 | 路径 / 值 |
|---|---|
| 仓库（GitHub） | git@github.com:Ericshunjian/ficc-dashboard.git |
| 网站 | https://ericshunjian.github.io/ficc-dashboard/ |
| 更新脚本 | daily_update.py |
| 路径常量位置 | daily_update.py line 28-41 |
| 数据源备份 | data_source/ |
| 研究结论云端文件 | research_conclusions.json（首次上传后生成） |
| Python | 3.13+，需 pandas + openpyxl |

---

## 十、常见问题

**Q: 网站数据没更新？**
A: GitHub Pages 有 1-2 分钟部署延迟；机构行为页用 IndexedDB 缓存，首次更新后需 Ctrl+F5 强制刷新。

**Q: 现券收益率漏最新一天？**
A: daily_update.py 首次运行时 pd.read_excel 可能读到公式缓存的旧值，导致最新一天被当 0 过滤。重新运行一次即可；务必验证 JSON 中各券最大日期 == 源文件最新有值日期。

**Q: data_source/ 的文件要更新吗？**
A: 不强制。它是换电脑初始化用的历史快照。日常更新用的是你每天下载的新文件（由 daily_update.py 路径常量指定）。若想让 data_source/ 也保持最新，可在每次更新后手动覆盖一次。
