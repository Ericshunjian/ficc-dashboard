# -*- coding: utf-8 -*-
"""一键更新 TS 蜘蛛网全部产物与「一页汇总」。

用法：
  python update_spider_all.py            # 源 Excel 有变化才重跑；无变化直接跳过
  python update_spider_all.py --force    # 忽略变化检测，强制全链路重跑
  python update_spider_all.py --check    # 只看源/产物状态，不跑任何脚本

检测范围：本目录下所有年份文件夹里的 Wind 会员持仓 xlsx + TS.CFE.xlsx。
执行顺序严格按 README 第三步 + 各独立课题 + 网格补算 + 汇总页生成。
产物只写本地；同步 FICC 主仓与 GitHub 推送由对话中的 SOP 完成。
"""
import argparse
import hashlib
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from spider_archive_rules import FEATURED

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, 'TS_更新状态.json')

# 顺序不能乱：② 的净值要由 ③ 回嵌进持仓整理网页
STEPS = [
    ('build_spider_web.py', '① 持仓整理 + 网页'),
    ('spider_backtest.py', '② 回测 + 回测网页'),
    ('build_spider_web.py', '③ 回嵌净值到持仓整理网页'),
    ('spider_rule_leverage_diag.py', '④ 规则 / 杠杆归因'),
    ('spider_pingan_signal.py', '⑤ 平安逆势加多检验'),
    ('spider_pingan_longonly.py', '⑥ 平安只做多'),
    ('spider_seat_compare.py', '⑦ 9 席位对比 + 中信归因'),
    ('spider_grid_detail.py', '⑧ 网格夏普王 / 点数王补算'),
    ('spider_stack_long.py', '⑨ 叠加仓位 vs 单仓位'),
    ('spider_hold_period.py', '⑩ 持有期 1/3/5/10 专项'),
    ('spider_summary.py', '⑪ 生成一页汇总'),
]

OUTPUTS = [
    'TS_持仓整理.json', 'TS_持仓整理_2021-2026.xlsx', 'TS_蜘蛛网_持仓整理.html',
    'TS_策略回测.json', 'TS_策略回测_交易明细.xlsx', 'TS_蜘蛛网_策略回测.html',
    'TS_规则杠杆归因.json',
    'TS_平安净多策略.json', 'TS_平安净多策略_明细.xlsx', 'TS_平安净多_策略.html',
    'TS_平安纯多头.json', 'TS_平安纯多头_明细.xlsx', 'TS_平安纯多头_策略.html',
    'TS_席位对比.json', 'TS_席位对比_明细.xlsx', 'TS_席位对比_净值.html',
    'TS_网格优选_明细.json', 'TS_叠加仓位.json', 'TS_持有期专项.json', 'index.html', 'TS_研究汇总.html', 'TS_每日信号.json', 'TS_每日信号.html', 'TS_前瞻记录.json',
]


def source_snapshot():
    """源 Excel 的 (相对路径, mtime_ns, size) 快照。"""
    files = glob.glob(os.path.join(BASE, '**', '*.xlsx'), recursive=True)
    out = {}
    for p in files:
        bn = os.path.basename(p)
        if bn.startswith('~$'):
            continue
        # 只认 Wind 会员持仓源文件与行情文件；所有 _明细.xlsx 都是产物
        if bn != 'TS.CFE.xlsx' and '会员：' not in bn:
            continue
        st = os.stat(p)
        out[os.path.relpath(p, BASE).replace('\\', '/')] = [st.st_mtime_ns, st.st_size]
    return out


def code_snapshot():
    files = sorted(glob.glob(os.path.join(BASE, '*.py')) +
                   glob.glob(os.path.join(BASE, '*_template.html')))
    return {os.path.basename(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}


def snapshot_summary(snap):
    holding = [k for k in snap if '会员：' in os.path.basename(k)]
    px = [k for k in snap if k.endswith('TS.CFE.xlsx')]
    return {'n_holding': len(holding), 'n_price': len(px),
            'latest': max((v[0] for v in snap.values()), default=0)}


def read_state():
    try:
        with open(STATE, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def verify():
    """核对末日一致、逐笔区块存在，返回摘要文本。"""
    msgs = []
    missing = [p for p in OUTPUTS if not os.path.exists(os.path.join(BASE, p))]
    if missing:
        return ['缺少产物: ' + ', '.join(missing)], False

    with open(os.path.join(BASE, 'TS_持仓整理.json'), encoding='utf-8') as f:
        holdings = json.load(f)
    with open(os.path.join(BASE, 'TS_每日信号.json'), encoding='utf-8') as f:
        daily = json.load(f)
    html = (Path(BASE) / 'index.html').read_text(encoding='utf-8')
    alias = (Path(BASE) / 'daily_signal_template.html').read_text(encoding='utf-8')
    if any((Path(BASE) / name).read_text(encoding='utf-8') != alias
           for name in ('TS_研究汇总.html', 'TS_每日信号.html')):
        return ['旧入口与每日信号跳转模板不同步'], False
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    if not m:
        return ['汇总页缺少数据'], False
    payload = json.loads(m.group(1))
    activity = daily.get('activity', {})
    legs = {leg['key'] for item in daily['strategies'] for leg in item.get('legs', [item])}
    quantiles = payload.get('pingan_quantiles', {})
    ok = (payload['asof'] == daily['asof'] and daily['asof'] >= holdings['dates'][-1]
          and [item.get('key') for item in daily['strategies']] == [rule.key for rule in FEATURED]
          and payload['strategies'] == daily['strategies']
          and activity == payload.get('activity')
          and activity.get('year') == daily['asof'][:4]
          and set(activity.get('books', {})) == legs
          and set(quantiles) == {'20', '75'}
          and all(len(levels) == len(payload['dates']) for levels in quantiles.values()))
    return [f"持仓快照 {holdings['dates'][-1]}，日更 {daily['asof']}，"
            f"{len(daily['strategies'])} 个策略"], ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='忽略源变化检测，强制重跑')
    ap.add_argument('--check', action='store_true', help='只检查状态，不执行')
    args = ap.parse_args()

    snap = source_snapshot()
    ss = snapshot_summary(snap)
    prev = read_state()
    source_code = code_snapshot()
    print('源 Excel: %d 张会员持仓 + %d 张行情；最新修改 %s' %
          (ss['n_holding'], ss['n_price'],
           time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ss['latest'] / 1e9)) if ss['latest'] else '无'))

    if args.check:
        changed = sorted(set(snap) ^ set(prev.get('sources', {}))) if prev else list(snap)
        mod = [k for k in snap if k in prev.get('sources', {}) and snap[k] != prev['sources'][k]] if prev else []
        print('新增/删除: %d；修改: %d' % (len(changed), len(mod)))
        for k in changed[:10] + mod[:10]:
            print('  ', k)
        lines, ok = verify()
        for line in lines:
            print(' ', line)
        return 0 if ok else 2

    if prev and prev.get('sources') == snap and prev.get('code') == source_code and not args.force:
        lines, ok = verify()
        if ok:
            print('输入和代码无变化，产物校验通过。')
            for line in lines:
                print(' ', line)
            return 0
        print('产物校验失败，重新生成。')

    if prev:
        diff = sorted(set(snap) ^ set(prev.get('sources', {})))
        mod = [k for k in snap if k in prev.get('sources', {}) and snap[k] != prev['sources'][k]]
        print('检测到变化：新增/删除 %d 张，修改 %d 张' % (len(diff), len(mod)))
        for k in diff[:12] + mod[:12]:
            print('  ', k)
    else:
        print('首次运行 / 无状态快照，执行全链路。')

    results = []
    for script, label in STEPS:
        t0 = time.time()
        p = subprocess.run([sys.executable, script], cwd=BASE,
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        tail = [x.strip() for x in p.stdout.strip().splitlines() if x.strip()][-3:]
        print('[%s] %-24s %5.1fs  %s' %
              ('OK' if p.returncode == 0 else 'FAIL', label, time.time() - t0, ' | '.join(tail)[:190]),
              flush=True)
        results.append({'script': script, 'rc': p.returncode, 'sec': round(time.time() - t0, 2)})
        if p.returncode != 0:
            print(p.stderr[-2000:], file=sys.stderr)
            print('中断：上一步失败，后续步骤未执行。', file=sys.stderr)
            return 1

    msgs, ok = verify()
    with open(STATE, 'w', encoding='utf-8', newline='\n') as f:
        json.dump({'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'sources': snap, 'code': source_code,
                   'steps': results, 'verify': msgs, 'ok': ok}, f, ensure_ascii=False, indent=1)
    print('--- 校验 ---')
    for line in msgs:
        print(' ', line)
    print('更新完成' if ok else '更新完成但校验有告警，请检查上表')
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
