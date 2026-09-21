# -*- coding: utf-8 -*-
"""
本地预览服务器 —— 在家里或其他电脑上打开 FICC 看板用。

为什么需要它：
    页面用 fetch() 读取同目录的 JSON。直接双击 HTML 会用 file:// 协议打开，
    浏览器的同源策略会拦掉 fetch，结果是白屏或图出不来。
    必须通过 http:// 访问，本脚本就是起这个 http 服务。

用法：
    python preview.py              # 默认端口 8000，自动打开首页
    python preview.py 8080         # 指定端口
    python preview.py 8000 factor_checkup.html   # 指定要打开的页面

    然后浏览器地址栏会出现 http://127.0.0.1:8000/index.html
    改完 HTML/JS 后按 Ctrl+F5 强制刷新即可看到效果。

只用 Python 标准库，无需安装任何包。Ctrl+C 退出。
"""

import http.server
import socketserver
import sys
import os
import time
import webbrowser
from functools import partial

BASE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PORT = 8000
DEFAULT_PAGE = "index.html"
MAX_PORT_TRIES = 10


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """常规静态服务，额外做两件事：关掉缓存以便改完即时生效、屏蔽访问日志噪音。"""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".json": "application/json; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
    }

    def end_headers(self):
        # 本地开发一律不缓存，避免改了 JS 刷新还是旧版
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def pick_port(start):
    for p in range(start, start + MAX_PORT_TRIES):
        if not port_in_use(p):
            return p
    raise SystemExit(f"端口 {start}~{start + MAX_PORT_TRIES - 1} 都被占用，换个起始端口再试")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    page = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PAGE

    if port_in_use(port):
        print(f"[提示] 端口 {port} 已被占用，往后找一个空端口")
        port = pick_port(port + 1)

    if not os.path.exists(os.path.join(BASE, page)):
        print(f"[警告] 找不到 {page}，改为打开首页")
        page = DEFAULT_PAGE

    handler = partial(QuietHandler, directory=BASE)
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/{page}"
        print("=" * 52)
        print("  FICC 看板本地预览已启动")
        print(f"  目录: {BASE}")
        print(f"  地址: {url}")
        print("  按 Ctrl+C 退出")
        print("=" * 52)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止预览服务")


if __name__ == "__main__":
    main()
