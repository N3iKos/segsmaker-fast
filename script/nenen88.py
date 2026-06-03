TOKET = ''
TOBRUT = ''

from IPython.core.magic import register_line_magic, register_cell_magic
from IPython.display import display, HTML, clear_output, Image
from urllib.parse import urlparse
from IPython import get_ipython
from pathlib import Path
from tqdm import tqdm
import ipywidgets as widgets
import subprocess
import threading
import requests
import zipfile
import shlex
import json
import time
import sys
import re
import os
import io

MAGENTA = '\033[35m'
RED = '\033[31m'
CYAN = '\033[36m'
GREEN = '\033[38;5;49m'
YELLOW = '\033[33m'
BLUE = '\033[38;5;69m'
PURPLE = '\033[38;5;177m'
ORANGE = '\033[38;5;208m'
RESET = '\033[0m'

CD = os.chdir
SyS = get_ipython().system
iRON = os.environ

KAGGLE = 'KAGGLE_DATA_PROXY_TOKEN' in iRON

CIVITAI = ['civitai.com', 'civitai.red']

@register_line_magic
def say(line):
    args = re.findall(r'\{[^\{\}]+\}|[^\s\{\}]+', line)
    output = []
    theme = get_ipython().config.get('InteractiveShellApp', {}).get('theme', 'light')
    default_color = 'white' if theme == 'dark' else 'black'

    i = 0
    while i < len(args):
        msg = args[i]
        color = None

        if re.match(r'^\{[^\{\}]+\}$', msg.lower()):
            color = msg[1:-1]
            msg = ''
        else:
            while i < len(args) - 1 and not re.match(r'^\{[^\{\}]+\}$', args[i + 1].lower()):
                i += 1
                msg += ' ' + args[i]

        if color == 'd':
            color = default_color
        elif color is None and i < len(args) - 1:
            if re.match(r'^\{[^\{\}]+\}$', args[i + 1].lower()):
                color = args[i + 1][1:-1]
                i += 1

        span_text = f'<span'
        if color:
            span_text += f" style='color:{color};'"
        span_text += f'>{msg}</span>'
        output.append(span_text)
        i += 1

    display(HTML(' '.join(output)))

@register_line_magic
def download(i):
    args = i.split()
    if not args:
        print('  missing URL, downloading nothing')
        return

    url = args[0]
    path = Path(url).expanduser()
    if url.endswith('.txt') and path.is_file():
        for l in path.read_text(encoding='utf-8').splitlines(): netorare(l)
    else: netorare(i)

def netorare(line, parallel=False):
    fp, fn = None, None

    parts = line.strip().split()
    if not parts: return

    cwd = Path.cwd()
    path = lambda s: '/' in s or '~/' in s
    url = parts[0].replace('\\', '')

    C = any(u in url for u in CIVITAI)
    H = 'huggingface.co' in url
    G = 'github.com' in url
    D = 'drive.google.com' in url

    if len(parts) >= 3:
        a, b = parts[1], parts[2]

        aa = path(a)
        bb = path(b)

        if bb and not aa: p, f = b, a
        elif aa and not bb: p, f = a, b
        elif Path(b).suffix == '' and Path(a).suffix != '': p, f = b, a
        else: p, f = a, b

        fp = Path(p).expanduser()
        fn = f

        fp.mkdir(parents=True, exist_ok=True)

    elif len(parts) == 2:
        a = parts[1]

        if path(a):
            fp = Path(a).expanduser()
            fp.mkdir(parents=True, exist_ok=True)
            fn = (None if (C or D) else Path(urlparse(url).path).name)
        else:
            fn = a
            fp = cwd

    else:
        fn = (None if (C or D) else Path(urlparse(url).path).name)
        fp = cwd

    if C or H or G: ariari(url, fp, fn, parallel=parallel)

    elif D: gdrown(url, fp, fn, parallel=parallel)

    else:
        cp = (len(parts) == 2 and fp is not None)
        cmd = (
          f"curl -#{'OJL' if len(parts) == 1 or cp else 'JL'} '{url}'" +
          (f" -o '{fn}'" if fn is not None and not cp else "")
        )
        curlly(cmd, fn, parallel=parallel, fp=fp)

def resizer(b, size=512):
    from PIL import Image
    i = Image.open(io.BytesIO(b))
    w, h = i.size
    s = (size, int(h * size / w)) if w > h else (int(w * size / h), size)
    o = io.BytesIO()
    i.resize(s, Image.LANCZOS).save(o, format='PNG')
    o.seek(0)
    return o

def get_civdom(url):
    try: return next((d for d in CIVITAI if d in urlparse(url).netloc.lower()), None)
    except: return None

def civitai_headers():
    return {'User-Agent': 'CivitaiLink:Automatic1111'}

def civitai_preview(j, p, fn, versionId=None):
    v = get_civitai(j, versionId)
    if not v: return

    images = v.get('images', [])
    name = fn or v.get('files', [{}])[0].get('name')
    if not name: return

    path = Path(p) / f'{Path(name).stem}.preview.png'
    if path.exists(): return

    preview = next((img.get('url', '') for img in images if not img.get('url', '').lower().endswith(('.mp4', '.gif'))), None)
    if not preview: return

    r = requests.get(preview, headers=civitai_headers()).content
    resized = resizer(r)

    if KAGGLE:
        from melon00 import image_encryption
        image_encryption(resized, path)
    else:
        path.write_bytes(resized.read())

def civitai_infotags(j, p, fn, versionId=None):
    v = get_civitai(j, versionId)
    if not v: return

    modelId = j.get('id') or v.get('modelId')
    name = fn or v.get('files', [{}])[0].get('name')
    if not name: return

    info = Path(p) / f'{Path(name).stem}.json'
    if info.exists(): return

    baseList = {
        'SD 1': 'SD1',
        'SD 1.5': 'SD1',
        'SD 2': 'SD2',
        'SD 3': 'SD3',
        'SDXL': 'SDXL',
        'Pony': 'SDXL',
        'Illustrious': 'SDXL',
        'Anima': 'Anima',
        'ZImageBase': 'ZImageBase',
        'ZImageTurbo': 'ZImageTurbo',
    }

    data = {
        'activation text': ', '.join(v.get('trainedWords', [])),
        'sd version': next((s for k, s in baseList.items() if k in v.get('baseModel', '')), ''),
        'modelId': modelId,
        'modelVersionId': v.get('id'),
        'sha256': v.get('files', [{}])[0].get('hashes', {}).get('SHA256')
    }

    info.write_text(json.dumps(data, indent=4))

def civitai_earlyAccess(j, versionId=None, civitai=None):
    v = get_civitai(j, versionId)
    if not v: return False

    if v.get('availability') == 'EarlyAccess' or v.get('earlyAccessEndsAt'):
        modelId = j.get('id') or v.get('modelId')
        modelVersionId = v.get('id')
        page = f'https://{civitai}/models/{modelId}?modelVersionId={modelVersionId}'
        print(f'{page}\n-> The model version is in early access and requires payment for downloading.')
        return True

    return False

def civitai_file(j, versionId=None):
    v = get_civitai(j, versionId)
    if not v: return None, None

    f = next((f for f in v.get('files', []) if f.get('downloadUrl')), None)
    n = ((f.get('name') if f else None) or v.get('name'))

    return f, n

def get_json(api_url, headers):
    try:
        r = requests.get(api_url, headers=headers, timeout=15)
        if r.status_code != 200: return None
        return r.json()
    except:
        return None

def get_civitai(j, versionId=None):
    v = None

    if versionId:
        if 'modelVersions' in j: v = next((mv for mv in j['modelVersions'] if str(mv.get('id')) == str(versionId)), None)
        if not v and str(j.get('id')) == str(versionId) and 'files' in j: v = j

    if not v:
        if 'modelVersions' in j: v = j['modelVersions'][0]
        else: v = j

    return v

def get_url(url, fn):
    civitai = get_civdom(url)

    if 'github.com' in url:
        return url.replace('/blob/', '/raw/'), None, None, fn

    elif 'huggingface.co' in url:
        url = url.split('?')[0]

        headers = {
            'User-Agent': 'Mozilla/5.0',
            **({'Authorization': f'Bearer {TOBRUT}'} if TOBRUT else {})
        }

        ext = ['.safetensors', '.pt', '.pth']
        j, versionId = None, None

        if fn and Path(fn).suffix.lower() in ext:
            try:
                raw_url = re.sub(r'/(resolve|blob)/', '/raw/', url)
                res = requests.get(raw_url, headers=headers, timeout=15)

                t = re.search(r'oid sha256:([a-fA-F0-9]{64})', res.text)
                if t:
                    sha256 = t.group(1).lower()

                    for c in CIVITAI:
                        try:
                            api_url = f'https://{c}/api/v1/model-versions/by-hash/{sha256}'
                            j_try = get_json(api_url, civitai_headers())
                            if not j_try: continue

                            r = next((f for f in j_try.get('files', []) if f.get('hashes', {}).get('SHA256', '').lower() == sha256), None)
                            if r:
                                j = j_try
                                break

                        except Exception: continue

            except Exception: pass

        url = url.replace('/blob/', '/resolve/')
        return url, j, versionId, fn

    elif civitai in url:
        input_url = url
        url = url.split('?token=')[0]

        if f'{civitai}/api/download/models/' in url:
            versionId = url.split('models/')[1].split('/')[0].split('?')[0]
            api_url = f'https://{civitai}/api/v1/model-versions/{versionId}'

            j = get_json(api_url, civitai_headers())
            if not j: return url, None, None, None

            f, cfn = civitai_file(j, versionId)
            if not f: return url, None, None, None

            return url, j, versionId, (fn or cfn)

        elif f'{civitai}/models/' in url:
            versionId = None
            modelId = url.split('models/')[1].split('/')[0].split('?')[0]

            if '?modelVersionId=' in url: versionId = url.split('?modelVersionId=')[1].split('&')[0]

            api_url = f'https://{civitai}/api/v1/models/{modelId}'
            j = get_json(api_url, civitai_headers())
            if not j or civitai_earlyAccess(j, versionId, civitai): return None, None, None, None

            f, cfn = civitai_file(j, versionId)
            if not f:
                print(f'Unable to find download URL for\n-> {input_url}\n')
                return None, None, None, None

            return f['downloadUrl'], j, versionId, (fn or cfn)

    return url, None, None, fn

def ariari(url, fp, fn, parallel=False):
    url, j, versionId, fn = get_url(url, fn)
    if not url: return

    civitai = get_civdom(url)
    target_name = fn or Path(urlparse(url).path).name or "unknown file"

    if parallel:
        print(f"  {PURPLE}●{RESET} {target_name} ▶ [Starting download...]")

    headers = {'User-Agent': (civitai_headers()['User-Agent'] if civitai else 'Mozilla/5.0')}

    if TOKET and f'{civitai}/api/download/models/' in url:
        headers['Authorization'] = f'Bearer {TOKET}'

        try:
            r = requests.get(url, headers=headers, allow_redirects=True, stream=True, timeout=30)
            if r.url and r.url != url: url = r.url
            r.close()

        except Exception as e:
            print(f'  Preflight failed: {e}')
            print('  Falling back to aria2 with Authorization header.')

    cmd = [
        'aria2c',
        f"--header=User-Agent: {headers['User-Agent']}",
        '--allow-overwrite=true', '--console-log-level=error', '--stderr=true',
        '-c', '-x16', '-s16', '-k1M', '-j5' 
    ]

    if TOBRUT and 'huggingface.co' in url: cmd.append(f'--header=Authorization: Bearer {TOBRUT}')
    if fn: cmd += ['-o', fn]

    cmd.append(url)

    try:
        p = subprocess.Popen(cmd, cwd=str(fp), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        aria2_output, bl, error_code, error_line = '', False, [], []

        while True:
            lines = p.stderr.readline()

            if lines == '' and p.poll() is not None: break

            if lines:
                aria2_output += lines

                for prog in lines.splitlines():
                    if 'errorCode' in prog or 'Exception' in prog:
                        error_code.append(prog)

                    if '|' in prog and 'error_line' in prog:
                        prog = re.sub(r'(\|\s*)(error_line)(\s*\|)', f'\\1{RED}\\2{RESET}\\3', prog)
                        first, _, last = prog.rpartition('|')
                        last = re.sub(r'/', f'{CYAN}/{RESET}', last)
                        prog = f'{first}|{last}'
                        error_line.append(prog)

                    m = re.match(
                        r'\[#\w+\s+'
                        r'(?:(\d+(?:\.\d+)?\w+/\d+(?:\.\d+)?\w+))?'
                        r'\((\d+%)\)'
                        r'.*?DL:(\d+(?:\.\d+)?\w+)'
                        r'(?:.*?ETA:(\d+\w+))?',
                        prog
                    )

                    if m and not parallel:
                        sizes, percent, speed, eta = m.groups()

                        percent = re.sub(r'(\d+)(%)', f'\\1{PURPLE}\\2{RESET}', percent)
                        parts = [f'{MAGENTA}({RESET}{percent}{MAGENTA}){RESET}']

                        if sizes:
                            current, total = sizes.split('/')
                            current = re.sub(r'(\d+(?:\.\d+)?)(\w+)', f'\\1{PURPLE}\\2{RESET}', current)
                            total = re.sub(r'(\d+(?:\.\d+)?)(\w+)', f'\\1{PURPLE}\\2{RESET}', total)
                            parts.append(f'{current}' f'{CYAN}/{RESET}' f'{total}')

                        speed = re.sub(r'(\d+(?:\.\d+)?)(\w+)', f'\\1{PURPLE}\\2{RESET}', speed)
                        parts.append(f'{CYAN}DL{RESET}:' f'{speed}')

                        if eta:
                            parts.append(f'{CYAN}ETA{RESET}:' f'{YELLOW}{eta}{RESET}')

                        body = ' '.join(parts)

                        r = (
                            f'{fn} '
                            #f'{MAGENTA}【{RESET}'
                            f'{body}'
                            #f'{MAGENTA}】{RESET}'
                        )

                        print(f"\r{' '*300}\r  {RED}●{RESET} {r}", end='')
                        sys.stdout.flush()

                        bl = True
                        break

        civitai = None
        error = error_code + error_line
        for lines in error: print(f'  {lines}')

        for lines in aria2_output.splitlines():
            if '|' in lines and 'OK' in lines:
                pipe = [p.strip() for p in lines.split('|')]

                if len(pipe) >= 4:
                    saved = pipe[3]
                    saved = re.sub(r'/', f'{ORANGE}/{RESET}', saved)
                    if parallel:
                        print(f"  {GREEN}●{RESET} {saved} ▶ [Completed]")
                    else:
                        print(f"\r{' '*300}\r  {GREEN}●{RESET} {saved}")
                    sys.stdout.flush()
                    bl = False

        bl and not parallel and print()
        p.wait()

        if j:
            civitai_infotags(j, fp, fn, versionId)
            threading.Thread(
                target=civitai_preview,
                args=(j, fp, fn, versionId),
                daemon=True
            ).start()

    except KeyboardInterrupt:
        print(f'\n{"":>2}^ Canceled')

def curlly(cmd, fn, parallel=False):
    try:
        p = subprocess.Popen(
            shlex.split(cmd), cwd=str(Path.cwd()),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )

        prog = re.compile(r'(\d+\.\d+)%')
        curl_output = ''

        if parallel:
            print(f"  {PURPLE}●{RESET} {fn} ▶ [Starting download...]")
            for line in iter(p.stderr.readline, ''):
                curl_output += line
            p.wait()
            if p.returncode == 0:
                print(f"  {GREEN}●{RESET} {fn} ▶ [Completed]")
        else:
            with tqdm(
                total=100, desc=f'{fn.ljust(58):>{58 + 2}}', initial=0,
                bar_format='{desc} 【{bar:20}】【{percentage:3.0f}%】',
                ascii='▷▶', file=sys.stdout
            ) as pbar:
                for line in iter(p.stderr.readline, ''):
                    if line.strip():
                        match = prog.search(line)
                        if match:
                            progress = float(match.group(1))
                            pbar.update(progress - pbar.n)
                            pbar.refresh()

                    curl_output += line
                pbar.close()
            p.wait()

        if p.returncode != 0:
            if 'curl: (23)' in curl_output:
                print(
                    f"{'':>2}^ File already exists; download skipped. "
                    "Append a custom name after the URL or PATH to overwrite."
                )
            elif 'curl: (3)' in curl_output:
                print('')
            else:
                print(f"{'':>2}^ Error: {curl_output}")
        else:
            pass

    except KeyboardInterrupt:
        print(f"{'':>2}^ Canceled")

def gdrown(url, fp=None, fn=None, parallel=False):
    folder = 'drive.google.com/drive/folders' in url
    cmd = ['gdown', '--fuzzy']

    if folder: cmd.append('--folder')
    cmd.append(url)

    name = fn or "Google Drive file"
    saved = None

    if fp:
        fp = Path(fp).expanduser()
        fp.mkdir(parents=True, exist_ok=True)

        if fn:
            fn = fp / fn
            cmd += ['-O', str(fn)]

        cwd = str(fp)

    else:
        cwd = None

        if fn: cmd += ['-O', fn]

    if parallel:
        print(f"  {PURPLE}●{RESET} {name} ▶ [Starting Google Drive download...]")

    try:
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        bl = False

        while True:
            prog = p.stdout.readline()

            if prog == '' and p.poll() is not None: break
            if not prog: continue

            prog = prog.strip()
            if not prog: continue

            if prog.startswith('To: '):
                try:
                    saved = prog[4:].strip()
                    name = Path(saved).name
                except: pass
                continue

            if '%' in prog and '/' in prog:
                if not parallel:
                    prog = re.sub(r'(\d+)(%)', f'\\1{PURPLE}\\2{RESET}', prog)
                    prog = re.sub(r'(\d+(?:\.\d+)?[KMG]B/s)', f'{CYAN}\\1{RESET}', prog)
                    print(f"\r{' '*300}\r  {RED}●{RESET} {name} {prog}", end='')

                    sys.stdout.flush()
                bl = True

            else:
                skip = (
                    'Downloading...' in prog or
                    'From (original):' in prog or
                    'From (redirected):' in prog
                )

                if skip: continue
                if bl and not parallel: print()

                if not parallel:
                    print(f'  {GREEN}●{RESET} {prog}')
                bl = False

        p.wait()

        if saved:
            saved = re.sub(r'/', f'{ORANGE}/{RESET}', saved)
            if parallel:
                print(f"  {GREEN}●{RESET} {saved} ▶ [Completed]")
            else:
                print(f"\r{' '*300}\r  {GREEN}●{RESET} {saved}")

    except KeyboardInterrupt:
        try: p.terminate()
        except: pass
        print(f'\n{"":>2}^ Canceled')

@register_line_magic
def clone(i):
    p = Path(i).expanduser()

    def proc(line):
        return line.strip()[len('git clone '):].strip() if line.strip().startswith('git clone') else line.strip()

    if p.suffix == '.txt' and p.is_file():
        cmds = [f'git clone {proc(line)}' for line in p.read_text().splitlines()]
    elif isinstance(i, str):
        cmds = [f'git clone {proc(i)}']
    else:
        cmds = [f'git clone {proc(l)}' for l in i]

    for cmd in cmds:
        cmd = cmd.strip()
        if not cmd:
            continue

        cmd_list = shlex.split(cmd)
        url = next((repo for repo in cmd_list if re.match(r'https?://', repo)), None)

        p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        while True:
            output = p.stdout.readline()
            if not output and p.poll() is not None:
                break

            if output := output.strip():
                if 'fatal' in output:
                    print(f'  {output}')
                elif output.startswith('Cloning into'):
                    repo_name = "/".join(output.split("'")[1].split("/")[-3:])
                    print(f'  {repo_name} ▶ {url}')

        p.wait()

@register_line_magic
def pull(line):
    inputs = line.split()
    if len(inputs) < 3: return

    subs = subprocess.run
    repo, tarfold, despath = inputs[:3]
    branch = inputs[3] if len(inputs) == 4 else None

    print(
        f"\n{'':>2}{'pull':<4} : {tarfold}",
        f"\n{'':>2}{'from':<4} : {repo}",
        f"\n{'':>2}{'into':<4} : {despath}",
        end=''
    )

    if branch: print(f"\n{'':>2}{'branch':<4} : {branch}")
    print('\n')

    fp = Path(despath).expanduser()
    opts = {'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE, 'check': True}
    cmd1 = f'git clone -n --depth=1 --filter=tree:0'
    if branch: cmd1 += f' --branch {branch}'
    cmd1 += f' {repo}'
    subs(shlex.split(cmd1), cwd=str(fp), **opts)

    repofold = fp / Path(repo).name.rstrip('.git')

    cmd2 = f'git sparse-checkout set --no-cone {tarfold}'
    subs(shlex.split(cmd2), cwd=str(repofold), **opts)

    cmd3 = 'git checkout'
    subs(shlex.split(cmd3), cwd=str(repofold), **opts)

    zipin = repofold / 'config' / tarfold
    zipout = fp / f'{tarfold}.zip'
    with zipfile.ZipFile(str(zipout), 'w') as zipf:
        for root in zipin.rglob('*'):
            if root.is_file():
                arcname = str(root.relative_to(zipin))
                zipf.write(str(root), arcname=arcname)

    cmd4 = f'unzip -o {str(zipout)}'
    subs(shlex.split(cmd4), cwd=str(fp), **opts)
    zipout.unlink()

    cmd5 = f'rm -rf {str(repofold)}'
    subs(shlex.split(cmd5), cwd=str(fp), **opts)

@register_line_magic
def tempe(line=''):
    try:
        from KANDANG import TEMPPATH
        TMP = Path(TEMPPATH)
    except ImportError:
        TMP = Path('/tmp')

    DIRS = [
        'ckpt',
        'lora',
        'controlnet',
        'svd',
        'z123',
        'clip',
        'clip_vision',
        'diffusers',
        'diffusion_models',
        'text_encoders',
        'unet'
    ]

    for SUB in DIRS: Path(f'{TMP}/{SUB}').mkdir(parents=True, exist_ok=True)

def parallel_download_files(lines, max_workers=3):
    from concurrent.futures import ThreadPoolExecutor
    lines = [l.strip() for l in lines if l.strip()]
    if not lines:
        print("No files to download.")
        return
    print(f"Downloading {len(lines)} files in parallel (Max workers: {max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(netorare, line, parallel=True) for line in lines]
        for f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"Error in parallel download thread: {e}")
    print("Parallel download batch complete.")

get_ipython().user_ns['parallel_download_files'] = parallel_download_files

@register_line_magic
def storage(line):
    import psutil
    from IPython.display import display, HTML
    
    home = Path.home()
    try:
        if os.path.exists('/content'):
            SyS("rm -rf /content/drive/MyDrive/.trash/* > /dev/null 2>&1")
        else:
            SyS(f"rm -rf {home}/.cache/* > /dev/null 2>&1")
    except:
        pass
    
    paths = ['/content', '/tmp'] if os.path.exists('/content') else [str(home), '/tmp']

    def size1(size, dcml=1):
        if size == 0: return '0 KB'
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                if unit in ['B', 'KB']: return f'{size:.0f} {unit}'
                else: return f'{size:.{dcml}f} {unit}'
            size /= 1024.0

    def size2(size_in_kb):
        if size_in_kb == 0: return '0 KB'
        base = 1024
        size_in_bytes = size_in_kb * base
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_in_bytes < base:
                if unit in ['B', 'KB']: return f'{size_in_bytes:.0f} {unit}'
                else: return f'{size_in_bytes:.1f} {unit}'
            size_in_bytes /= base

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            usage = psutil.disk_usage(path_str)
            size_str = size1(usage.total, dcml=0)
            used_str = size1(usage.used, dcml=1)
            free_str = size1(usage.free, dcml=1)

            if path_str in ['/content', str(home)]:
                storage_type = 'Persistent Storage'
            elif path_str == '/tmp':
                storage_type = 'Temporary Storage'
            else:
                storage_type = f'Storage ({path_str})'

            display(HTML(f'<b>{storage_type}</b>'))
            print(f' Size = {size_str:>8}')
            print(f' Used = {used_str:>8} | {usage.percent:.1f}%')
            print(f' Free = {free_str:>8} | {100 - usage.percent:.1f}%')
            print()
        except Exception as e:
            print(f"Error checking storage for {path_str}: {e}")

    du_target = '/content' if os.path.exists('/content') else str(home)
    try:
        du_process = subprocess.Popen(['du', '-h', '-k', '--max-depth=1', du_target], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        du_output = du_process.communicate()[0].decode()
        lines = du_output.split('\n')
        sub_paths = [Path(line.split('\t')[1]) for line in lines if line and '\t' in line]
        sizes_kb = [int(line.split('\t')[0]) for line in lines if line and '\t' in line]

        subdirectories = []
        for sub_path, size_kb in zip(sub_paths, sizes_kb):
            formatted_size = size2(size_kb)
            base_path = sub_path.name
            if base_path and base_path not in ['studio-lab-user', 'content']:
                subdirectories.append((base_path, formatted_size))

        if subdirectories:
            print("Directory Sizes:")
            for base_path, formatted_size in subdirectories:
                padding = ' ' * max(0, 9 - len(formatted_size))
                print(f'/{base_path:<30} {padding}{formatted_size}')
    except:
        pass

@register_cell_magic
def zipping(line, cell):
    lines = cell.strip().split('\n')

    input_path = None
    output_path = None
    custom_name = None

    for line in lines:
        soup = line.split('=')

        if len(soup) == 2:
            arg_name = soup[0].strip()
            arg_value = soup[1].strip().strip('"').strip("'")

            if '$HOME' in arg_value or '$home' in arg_value:
                arg_value = arg_value.replace('$HOME', str(Path.home())).replace('$home', str(Path.home()))

            if arg_value.startswith('$'):
                var_name = arg_value[1:]
                if var_name in get_ipython().user_ns:
                    arg_value = str(get_ipython().user_ns[var_name])
                elif var_name in globals():
                    arg_value = str(globals()[var_name])
                else:
                    print(f'[ERROR]: {var_name} is not defined.')
                    return

            if arg_name == 'inputs':
                input_path = Path(arg_value)

            elif arg_name == 'outputs':
                output_path = Path(arg_value)

            elif arg_name == 'name':
                custom_name = arg_value

    if not input_path or not input_path.exists():
        print(f'[ERROR]: {input_path} does not exist.')
        return

    if not output_path:
        output_path = Path.cwd()

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    def zip_folder(input_path, output_path, max_size_mb=20, custom_name=None):
        all_files = []
        skip_extensions = {
            '.safetensors', '.ckpt', '.pt', '.pth',
            '.h5', '.pickle', '.pkl', '.bin',
            '.zip', '.tar.gz', '.tar.lz4', '.py'
        }

        for file_path in input_path.rglob('*'):
            if file_path.is_file():
                if file_path.suffix.lower() not in skip_extensions:
                    all_files.append(file_path)
                else:
                    print(f'{file_path.name} skipped')

        zip_number = 1
        current_zip_size = 0

        if custom_name:
            current_zip_name = output_path / f'{custom_name}_{zip_number}.zip'
        else:
            current_zip_name = output_path / f'part_{zip_number}.zip'

        with tqdm(
            total=len(all_files),
            desc='zipping : ',
            bar_format='{desc}[{bar:26}] [{n_fmt}/{total_fmt}]',
            ascii='▷▶',
            file=sys.stdout
        ) as pbar:

            with zipfile.ZipFile(current_zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as current_zip:
                for file_path in all_files:
                    file_size = file_path.stat().st_size

                    if current_zip_size + file_size > max_size_mb * 1024 * 1024:
                        current_zip.close()
                        zip_number += 1

                        if custom_name:
                            current_zip_name = output_path / f'{custom_name}_{zip_number}.zip'
                        else:
                            current_zip_name = output_path / f'part_{zip_number}.zip'

                        current_zip = zipfile.ZipFile(current_zip_name, 'w', compression=zipfile.ZIP_DEFLATED)
                        current_zip_size = 0

                    current_zip.write(file_path, file_path.relative_to(input_path))
                    current_zip_size += file_size
                    pbar.update(1)

    max_size_mb = 200
    zip_folder(input_path, output_path, max_size_mb, custom_name)

@register_line_magic
def change_key(line):
    home = Path.home()
    startup_dir = home / '.ipython/profile_default/startup'
    nenen = startup_dir / 'nenen88.py'
    src = home / 'gutris1' if os.path.exists('/content') else home / '.gutris1'
    key_file = src / 'api-key.json'
    css = src / 'segsmaker.css' if not os.path.exists('/content') else None

    main_output = widgets.Output()
    save_button = widgets.Button(description='Save')
    cancel_button = widgets.Button(description='Cancel')
    new_civitai_key = widgets.Text(placeholder='New Civitai API KEY')
    new_hf_token = widgets.Text(placeholder='New Huggingface READ Token (optional)', layout=widgets.Layout(left='6px', width='340px'))
    current_civitai_key = widgets.Text(placeholder='', disabled=True, layout=widgets.Layout(left='-3px', top='0px'))
    current_hf_token = widgets.Text(placeholder='', disabled=True, layout=widgets.Layout(top='0px'))

    buttons = widgets.HBox(
        [cancel_button, save_button],
        layout=widgets.Layout(
            width='400px',
            display='flex',
            flex_flow='row',
            align_items='center',
            justify_content='space-around',
            padding='0px'
        )
    )

    current_box = widgets.Box(
        [current_civitai_key, current_hf_token],
        layout=widgets.Layout(
            left='12px',
            top='0px',
            height='100px',
            display='flex',
            flex_flow='column',
            justify_content='space-around',
            align_items='baseline'
        )
    )

    new_box = widgets.Box(
        [new_civitai_key, new_hf_token],
        layout=widgets.Layout(
            left='50px',
            top='-10px',
            height='100px',
            display='flex',
            flex_flow='column',
            justify_content='space-around',
            align_items='flex-end'
        )
    )

    key_box = widgets.HBox([current_box, new_box])

    input_widget = widgets.VBox(
        [key_box, buttons],
        layout=widgets.Layout(
            position="absolute",
            width="700px",
            height="180px",
            display="flex",
            flex_flow="column",
            align_items="center",
            justify_content="space-around",
            padding="20px",
        )
    )

    save_button.add_class('save')
    cancel_button.add_class('cancel')
    new_civitai_key.add_class('key-input')
    new_hf_token.add_class('key-hf')
    current_civitai_key.add_class('current-key')
    current_hf_token.add_class('current-hf')
    input_widget.add_class('input-widget')

    def key_inject(civitai_key, hf_token):
        if os.path.exists('/content'):
            SyS(f'curl -sLo {nenen} https://github.com/N3iKos/segsmaker-fast/raw/main/script/nenen88.py')
        p = Path(nenen)
        if p.exists():
            v = p.read_text()
            v = v.replace("TOKET = ''", f"TOKET = '{civitai_key}'")
            v = v.replace("TOBRUT = ''", f"TOBRUT = '{hf_token}'")
            p.write_text(v)

    def key_widget(current_civitai_key_value='', current_hf_token_value=''):
        current_civitai_key.value = current_civitai_key_value
        current_hf_token.value = current_hf_token_value

        def save_key(b):
            civitai_key = new_civitai_key.value.strip()
            hf_token = new_hf_token.value.strip()

            with main_output:
                if not civitai_key:
                    print('Please enter your CivitAI API Key')
                    return

                if len(civitai_key) < 32:
                    print('API key must be at least 32 characters long')
                    return

                civitai_ke = {'civitai-api-key': civitai_key}
                hf_toke = {'huggingface-read-token': hf_token}

                secrets = {**civitai_ke, **hf_toke}
                src.mkdir(parents=True, exist_ok=True)
                key_file.write_text(json.dumps(secrets, indent=4))

            with main_output:
                input_widget.close()
                main_output.clear_output(wait=True)
                say('Saving...')
                key_inject(civitai_key, hf_token)

                main_output.clear_output(wait=True)
                get_ipython().kernel.do_shutdown(True)
                time.sleep(2)
                say('Kernel restarting...')

                main_output.clear_output(wait=True)
                time.sleep(3)
                say('Done')

        def cancel_key(b):
            new_civitai_key.value = ''
            new_hf_token.value = ''

            with main_output:
                input_widget.close()
                main_output.clear_output(wait=True)
                say('^ Canceled')

        save_button.on_click(save_key)
        cancel_button.on_click(cancel_key)

    def key_check():
        if key_file.exists():
            try:
                v = json.loads(key_file.read_text())
                civitai_key = v.get('civitai-api-key', '')
                hf_token = v.get('huggingface-read-token', '')
                key_widget(civitai_key, hf_token)
            except:
                key_widget('', '')
            display(input_widget, main_output)
        else:
            key_widget('', '')
            display(input_widget, main_output)

    if css and css.exists():
        display(HTML(f'<style>{css.read_text()}</style>'))
    key_check()

@register_line_magic
def zrok_register(line):
    home = Path.home()
    zrok_bin = home / '.zrok/bin'
    zrok_cmd = zrok_bin / 'zrok invite'
    zrok_txt = zrok_bin / 'zrok_log.txt'

    zrok_output = widgets.Output()
    register_button = widgets.Button(description='Register', layout=widgets.Layout(left= '-45%'))
    exit_button = widgets.Button(description='Exit', layout=widgets.Layout(left= '45%'))
    email_input = widgets.Text(placeholder='Enter Your Valid Email Address', layout=widgets.Layout(width= '75%'))

    zrok_button = widgets.HBox(
        [register_button, exit_button],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='space-between'
        )
    )

    zrok_widget = widgets.VBox(
        [email_input, zrok_button],
        layout=widgets.Layout(
            height='160px',
            width= '550px',
            display='flex',
            flex_flow='column',
            align_items='center',
            justify_content='space-around',
            padding='20px'
        )
    )

    register_button.add_class('zrok-btn')
    exit_button.add_class('zrok-btn')
    email_input.add_class('email-input')
    zrok_widget.add_class('zrok-widget')

    def zrok_install():    
        if zrok_bin.exists(): return

        zrok_bin.mkdir(parents=True, exist_ok=True)
        zrok_url = 'https://github.com/openziti/zrok/releases/download/v1.0.2/zrok_1.0.2_linux_amd64.tar.gz'
        zrok_tar = zrok_bin / Path(zrok_url).name

        SyS(f'curl -sLo {zrok_tar} {zrok_url}')
        SyS(f'tar -xzf {zrok_tar} -C {zrok_bin} --wildcards *zrok')
        SyS(f'rm -rf {home}/.cache/* {zrok_tar}')

    def register(b):
        import pexpect

        zrok_widget.close()
        email = email_input.value

        R = '\033[0m'
        O = '\033[38;5;208m'
        E = f'{O}{email}{R}'

        with zrok_output:
            if not email:
                print('No email address entered.')
                return

            print('Submitting...')
            clear_output(wait=True)

            zrok_txt.touch()

            child = pexpect.spawn('bash')
            child.sendline(f'{zrok_cmd} | tee {zrok_txt}')
            child.expect('enter and confirm your email address...')

            for _ in range(2):
                time.sleep(1)
                child.sendline(email)
                time.sleep(1)
                child.send(chr(9))

            child.sendline('\r\n')
            time.sleep(2)
            child.close()

            print(f'Invitation sent to {E}\n Be sure to check your SPAM folder if you do not receive the invitation email.')
            try: zrok_txt.unlink()
            except: pass

    def exit(b):
        zrok_widget.close()

    src = home / 'gutris1' if os.path.exists('/content') else home / '.gutris1'
    css = src / 'segsmaker.css' if not os.path.exists('/content') else None
    if css and css.exists():
        display(HTML(f'<style>{css.read_text()}</style>'))
        
    display(zrok_widget, zrok_output)

    register_button.on_click(register)
    exit_button.on_click(exit)

    zrok_install()
    SyS('pip install -q pexpect')

@register_line_magic
def uninstall_webui(line):
    try:
        home = Path.home()
        marked = home / 'gutris1/marking.json' if os.path.exists('/content') else home / '.gutris1/marking.json'
        if marked.exists():
            config = json.loads(marked.read_text())
            ui = config.get('ui')
            if ui:
                webui_path = Path('/content') / ui if os.path.exists('/content') else home / ui
                if webui_path.exists():
                    import shutil
                    shutil.rmtree(webui_path, ignore_errors=True)
                    print(f"{ui} has been uninstalled.")
                else:
                    print(f"{ui} path not found: {webui_path}")
            else:
                print("No UI active in marking.json")
        else:
            print("marking.json not found")
    except Exception as e:
        print(f"Error uninstalling: {e}")

@register_line_magic
def delete_everything(line):
    import shutil
    home = Path.home()
    print("Deleting installed components...")
    folder_list = [
        'A1111', 'Forge', 'ReForge', 'Forge-Classic', 'ComfyUI', 'SwarmUI', 'SDTrainer', 'FaceFusion',
        'tmp/*', 'tmp', '.cache/*', '.config/*', '.ssh', '.zrok', '.ngrok', '.sagemaker',
        '.conda/*', '.conda', '.ipython/profile_default/startup/*'
    ]
    if os.path.exists('/content'):
        colab_folders = ['A1111', 'Forge', 'ReForge', 'ReForge-old', 'Forge-Classic', 'Forge-Neo', 'ComfyUI', 'SwarmUI']
        for folder in colab_folders:
            p = Path('/content') / folder
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
        shutil.rmtree('/content/temp', ignore_errors=True)
    
    for f in folder_list:
        p = home / f
        SyS(f"rm -rf {p} >/dev/null 2>&1")
    print("Cleanup completed.")
