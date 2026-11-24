# -*- coding: utf-8 -*-
"""把 TS_每日信号.json 注入单文件模板，生成可离线打开的 TS_每日信号.html。"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(BASE, 'daily_signal_template.html')
DATA = os.path.join(BASE, 'TS_每日信号.json')
OUT = os.path.join(BASE, 'TS_每日信号.html')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def main():
    payload = open(DATA, encoding='utf-8').read()
    tpl = open(TPL, encoding='utf-8').read().replace('\r\n', '\n')
    if 'const D=__PAYLOAD__;' not in tpl:
        raise SystemExit('模板缺少 __PAYLOAD__ 占位符')
    html = tpl.replace('const D=__PAYLOAD__;', 'const D=' + payload + ';')
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write(html)
    print('HTML:', OUT, len(html), 'chars /', os.path.getsize(OUT), 'bytes')


if __name__ == '__main__':
    main()
