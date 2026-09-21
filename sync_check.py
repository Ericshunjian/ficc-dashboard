# -*- coding: utf-8 -*-
"""
开工自检 —— 在家里的电脑上改页面之前先跑它。

它会告诉你三件事：
    1. 本地 Git 设置对不对（core.autocrlf 必须是 false，否则会冒出大量假改动）
    2. 本地有没有落后于 GitHub（落后了直接改，最后 push 会被拒绝）
    3. 有没有上次没提交干净的改动

用法：
    python sync_check.py            # 只检查，报告情况
    python sync_check.py pull       # 检查并自动快进到最新（推荐开工就用这个）

只用 Python 标准库。
"""

import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))
REMOTE = "github"
BRANCH = "main"

OK, WARN, BAD = "  [OK]  ", "  [注意] ", "  [警告] "


def run(args, check=False):
    return subprocess.run(
        args, cwd=BASE, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=check
    )


def sh(args):
    r = run(args)
    return (r.stdout or "").strip() if r.returncode == 0 else ""


def main():
    do_pull = len(sys.argv) > 1 and sys.argv[1] == "pull"
    print("=" * 56)
    print("  FICC 看板 开工自检")
    print("=" * 56)

    problems = 0

    # 1. 身份与换行符设置
    user = sh(["git", "config", "user.name"])
    email = sh(["git", "config", "user.email"])
    crlf = sh(["git", "config", "core.autocrlf"]) or "(未设置)"

    if user and email:
        print(f"{OK}Git 身份:  {user} <{email}>")
    else:
        print(f"{BAD}Git 身份没配，commit 会用错的名字")
        print("        修: git config user.name \"你的名字\"")
        print("        修: git config user.email \"你的邮箱\"")
        problems += 1

    if (crlf or "").lower() == "false":
        print(f"{OK}core.autocrlf = false（正确，不会产生行尾假改动）")
    else:
        print(f"{BAD}core.autocrlf = {crlf}，应为 false")
        print("        不修的话，git status 会显示一堆没真改过的文件")
        run(["git", "config", "core.autocrlf", "false"])
        print("        已自动改成 false")
        problems += 1

    # 2. 抓取远程，比对落后多少
    print()
    fetch = run(["git", "fetch", REMOTE, BRANCH])
    if fetch.returncode != 0:
        print(f"{BAD}连不上 GitHub")
        print(f"        {fetch.stderr.strip()[:200]}")
        print("        常见原因: 没联网 / SSH key 没加到 GitHub 账号")
        problems += 1
        return 1

    head = sh(["git", "rev-parse", "HEAD"])
    remote_sha = sh(["git", "rev-parse", "FETCH_HEAD"])
    base = sh(["git", "merge-base", "HEAD", "FETCH_HEAD"])

    behind = sh(["git", "rev-list", "--count", f"{head}..FETCH_HEAD"]) if base else "0"
    ahead = sh(["git", "rev-list", "--count", f"FETCH_HEAD..{head}"]) if base else "0"
    behind_n = int(behind or 0)
    ahead_n = int(ahead or 0)

    if behind_n == 0 and ahead_n == 0:
        print(f"{OK}和 GitHub 完全一致，可以开始改")
    elif behind_n > 0 and ahead_n == 0:
        print(f"{WARN}本地落后 {behind_n} 个提交（办公室电脑更新过数据）")
        if do_pull:
            ff = run(["git", "merge", "--ff-only", "FETCH_HEAD"])
            if ff.returncode == 0:
                print(f"{OK}已快进到最新")
            else:
                print(f"{BAD}自动快进失败，需要人工处理")
                print(f"        {(ff.stderr or ff.stdout).strip()[:300]}")
                problems += 1
        else:
            print("        开工前执行: python sync_check.py pull")
            problems += 1
    elif behind_n == 0 and ahead_n > 0:
        print(f"{WARN}本地领先 {ahead_n} 个提交 —— 上次改完忘了 push？")
        print("        收工时执行: git push github main")
    else:
        print(f"{BAD}本地和远程分叉了（领先 {ahead_n} / 落后 {behind_n}）")
        print("        两边改了同一个东西。别自己 merge，先把改动打包发给我处理。")
        problems += 1

    # 3. 未提交的改动
    print()
    porcelain = sh(["git", "status", "--porcelain"])
    if porcelain:
        lines = [ln for ln in porcelain.splitlines() if ln.strip()]
        print(f"{WARN}有 {len(lines)} 个文件没提交:")
        for ln in lines[:10]:
            print(f"        {ln}")
        if len(lines) > 10:
            print(f"        ... 还有 {len(lines) - 10} 个")
    else:
        print(f"{OK}工作区干净，没有遗留改动")

    print()
    print("=" * 56)
    if problems == 0:
        print("  可以开始改页面了")
        print("  预览: python preview.py")
    else:
        print(f"  有 {problems} 项需要先处理，看上面的提示")
    print("=" * 56)
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
