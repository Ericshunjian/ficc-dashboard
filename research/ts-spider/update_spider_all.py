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
import glob
import json
import os
import re
import subprocess
import sys
import time

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
    ('spider_summary.py', '⑨ 生成一页汇总'),
]

OUTPUTS = [
    'TS_持仓整理.json', 'TS_持仓整理_2021-2026.xlsx', 'TS_蜘蛛网_持仓整理.html',
    'TS_策略回测.json', 'TS_策略回测_交易明细.xlsx', 'TS_蜘蛛网_策略回测.html',
    'TS_规则杠杆归因.json',
    'TS_平安净多策略.json', 'TS_平安净多策略_明细.xlsx', 'TS_平安净多_策略.html',
    'TS_平安纯多头.json', 'TS_平安纯多头_明细.xlsx', 'TS_平安纯多头_策略.html',
    'TS_席位对比.json', 'TS_席位对比_明细.xlsx', 'TS_席位对比_净值.html',
    'TS_网格优选_明细.json', 'TS_研究汇总.html',
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

    with open(os.path.join(BASE, 'TS_席位对比.json'), encoding='utf-8') as f:
        seat = json.load(f)
    with open(os.path.join(BASE, 'TS_平安纯多头.json'), encoding='utf-8') as f:
        pure = json.load(f)
    with open(os.path.join(BASE, 'TS_研究汇总.html'), encoding='utf-8') as f:
        html = f.read()
    m = re.search(r'const D=(\{.*?\});</script>', html, re.S)
    if not m:
        return ['汇总页 payload 解析失败'], False
    payload = json.loads(m.group(1))
    ty = payload.get('trades_year', {})
    ends = {
        '席位对比': seat['meta']['end'],
        '平安纯多': pure['meta']['end'] if 'meta' in pure else pure['preset'].get('end', ''),
        '汇总页': payload['meta']['end'],
        '逐笔截至': ty.get('asof', ''),
    }
    ok = len(set(v for v in ends.values() if v)) == 1 and '今年以来的开平仓记录' in html
    msgs.append('数据末日: ' + ', '.join('%s=%s' % kv for kv in ends.items()))
    msgs.append('逐笔: %s 年 / %d 个策略 / %d 笔' %
                (ty.get('year', '?'), len(ty.get('strategies', [])),
                 sum(s['n'] for s in ty.get('strategies', []))))
    msgs.append('汇总页: %d 字符 / %d 条净值 / %d 行指标' %
                (len(html), len(payload.get('series', {})), len(payload.get('rows', []))))
    return msgs, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='忽略源变化检测，强制重跑')
    ap.add_argument('--check', action='store_true', help='只检查状态，不执行')
    args = ap.parse_args()

    snap = source_snapshot()
    ss = snapshot_summary(snap)
    prev = read_state()
    print('源 Excel: %d 张会员持仓 + %d 张行情；最新修改 %s' %
          (ss['n_holding'], ss['n_price'],
           time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ss['latest'] / 1e9)) if ss['latest'] else '无'))

    if args.check:
        changed = sorted(set(snap) ^ set(prev.get('sources', {}))) if prev else list(snap)
        mod = [k for k in snap if k in prev.get('sources', {}) and snap[k] != prev['sources'][k]] if prev else []
        print('新增/删除: %d；修改: %d' % (len(changed), len(mod)))
        for k in changed[:10] + mod[:10]:
            print('  ', k)
        lines, _ = verify()
        for line in lines:
            print(' ', line)
        return 0

    if prev and prev.get('sources') == snap and not args.force:
        print('源 Excel 无变化，跳过重跑。需强制重跑请加 --force。')
        lines, _ = verify()
        for line in lines:
            print(' ', line)
        return 0

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
        json.dump({'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'sources': snap,
                   'steps': results, 'verify': msgs, 'ok': ok}, f, ensure_ascii=False, indent=1)
    print('--- 校验 ---')
    for line in msgs:
        print(' ', line)
    print('更新完成' if ok else '更新完成但校验有告警，请检查上表')
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
