import json
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
import urllib.parse

def update_config_in_file(file_path, config_key, new_config):
    """
    在指定的 Python 脚本文件中直接定位并替换对应配置键的字典内容。
    通过简单的字符串匹配和括号计数来实现将前端保存的新配置覆盖原代码中的配置。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 定位字典键所在位置（兼容双引号和单引号）
    start_idx = content.find(f'"{config_key}":')
    if start_idx == -1:
        start_idx = content.find(f"'{config_key}':")
        
    if start_idx == -1:
        return False
        
    # 找到字典的左大括号
    brace_start = content.find('{', start_idx)
    if brace_start == -1:
        return False
        
    # 通过括号计数找到对应字典的右大括号，以确定字典代码的范围
    brace_count = 0
    brace_end = -1
    for i in range(brace_start, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                brace_end = i
                break
                
    if brace_end == -1:
        return False
        
    # 将新的配置字典格式化为 JSON 字符串
    dict_str = json.dumps(new_config, indent=4)
    
    # 获取原始代码中字典的缩进，以保持代码对齐
    line_start = content.rfind('\n', 0, start_idx)
    if line_start == -1: line_start = 0
    indent = start_idx - line_start - 1
    if indent < 0: indent = 4
    spaces = " " * indent
    
    # 为多行配置添加相应的缩进
    formatted_lines = []
    for i, line in enumerate(dict_str.split('\n')):
        if i == 0:
            formatted_lines.append(line) 
        else:
            formatted_lines.append(spaces + line)
    
    # 修复 JSON 布尔值转换为 Python 布尔值，以直接替换原脚本
    formatted_str = '\n'.join(formatted_lines).replace(': true,', ': True,').replace(': false,', ': False,')
    
    # 将新的字典内容写入原脚本文件中的对应位置
    new_content = content[:brace_start] + formatted_str + content[brace_end+1:]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return True

PORT = 8085
HTML_PATH = '/scripts/watermark_ui.html'

class Handler(SimpleHTTPRequestHandler):
    """
    处理前端 HTTP 请求的简易服务器处理器：
    - GET /              → 302 跳转到 HTML 页面
    - GET 其他路径       → SimpleHTTPRequestHandler 静态文件服务
    - POST /api/save_config → 解析配置并写回 add_lane_watermarks.py
    """

    def _send_cors_headers(self):
        """所有响应统一附加 CORS 头，避免浏览器跨域拦截。"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        """处理预检请求（浏览器在跨域 POST 前自动发送）。"""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        # 根路径直接跳转到 HTML 页面，避免用户手动拼 URL
        if self.path in ('/', ''):
            self.send_response(302)
            self.send_header('Location', HTML_PATH)
            self._send_cors_headers()
            self.end_headers()
        elif self.path.startswith('/api/get_config'):
            import ast
            try:
                script_path = os.path.join(os.path.dirname(__file__), 'add_lane_watermarks.py')
                with open(script_path, 'r', encoding='utf-8') as sf:
                    content = sf.read()
                start_idx = content.find("WATERMARK_CONFIGS = {")
                if start_idx != -1:
                    brace_start = content.find('{', start_idx)
                    brace_count = 0
                    brace_end = -1
                    for i in range(brace_start, len(content)):
                        if content[i] == '{': brace_count += 1
                        elif content[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                brace_end = i
                                break
                    if brace_end != -1:
                        dict_str = content[brace_start:brace_end+1]
                        config_dict = ast.literal_eval(dict_str)
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self._send_cors_headers()
                        self.end_headers()
                        self.wfile.write(json.dumps(config_dict).encode())
                        return
            except Exception as e:
                print(f"Error reading config: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b"{}")
        else:
            # 其余路径交由父类提供静态文件服务（CSS、JS、图片等）
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/save_config':
            content_len = int(self.headers.get('Content-Length', 0))
            post_body   = self.rfile.read(content_len)
            try:
                data       = json.loads(post_body)
                config_key = data['config_key']
                config     = data['config']

                script_path = os.path.join(os.path.dirname(__file__), 'add_lane_watermarks.py')
                success     = update_config_in_file(script_path, config_key, config)

                self.send_response(200 if success else 500)
                self.send_header('Content-Type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'success': success}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """屏蔽静态文件的日志噪音，只打印 API 请求。"""
        if hasattr(self, 'path') and self.path and '/api/' in self.path:
            super().log_message(format, *args)


if __name__ == '__main__':
    # 切换到项目根目录（scripts/ 的上级），使静态文件服务能覆盖 data/ 和 scripts/
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    server_address = ('0.0.0.0', PORT)
    httpd = HTTPServer(server_address, Handler)

    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    print('=' * 55)
    print(f'  Watermark Tuner Server started on port {PORT}')
    print(f'  本地访问: http://localhost:{PORT}')
    print(f'  局域网访问: http://{local_ip}:{PORT}')
    print(f'  页面路径: http://localhost:{PORT}{HTML_PATH}')
    print('  按 Ctrl+C 停止服务')
    print('=' * 55)
    httpd.serve_forever()
