TOKET = ''
TOBRUT = ''

from IPython.core.magic import register_line_magic
from IPython.display import display, HTML, clear_output
from urllib.parse import urlparse
from IPython import get_ipython
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from threading import Lock
import subprocess
import requests
import zipfile
import shlex
import json
import sys
import re
import os
import io

MAGENTA = '\033[35m'
RED = '\033[31m'
CYAN = '\033[36m'
GREEN = '\033[38;5;35m'
YELLOW = '\033[33m'
BLUE = '\033[38;5;69m'
PURPLE = '\033[38;5;135m'
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

def netorare(line):
    fp, fn = None, None

    parts = line.strip().split()
    if not parts: return

    cwd = Path.cwd()
    url = parts[0].replace('\\', '')
    CHG = any(domain in url for domain in [*CIVITAI, 'huggingface.co', 'github.com'])
    DriveGoogle = 'drive.google.com' in url

    path = lambda s: '/' in s or '~/' in s

    try:
        if len(parts) >= 3:
            arg1, arg2 = parts[1], parts[2]
            path_arg, file_arg = (arg2, arg1) if path(arg2) and not path(arg1) else \
                                 (arg1, arg2) if path(arg1) and not path(arg2) else \
                                 (arg2, arg1) if Path(arg2).suffix == '' and Path(arg1).suffix != '' else \
                                 (arg1, arg2)

            fp, fn = Path(path_arg).expanduser(), file_arg
            fp.mkdir(parents=True, exist_ok=True)
            CD(fp)

        elif len(parts) == 2:
            arg = parts[1]
            if path(arg):
                fp = Path(arg).expanduser()
                fp.mkdir(parents=True, exist_ok=True)
                CD(fp)
                fn = get_fn(url) if CHG else Path(urlparse(url).path).name
            else:
                fn = arg
                fp = cwd
        else:
            fn = get_fn(url) if CHG else Path(urlparse(url).path).name
            fp = cwd

        if CHG: ariari(url, fp, fn)
        elif DriveGoogle: gdrown(url, fp, fn)
        else:
            path_only = len(parts) == 2 and fp is not None
            cmd = f"curl -#{'OJL' if len(parts) == 1 or path_only else 'JL'} '{url}'" + (f" -o '{fn}'" if fn is not None and not path_only else "")
            curlly(cmd, fn)
    finally:
        CD(cwd)

def resizer(b, size=512):
    from PIL import Image
    i = Image.open(io.BytesIO(b))
    w, h = i.size
    s = (size, int(h * size / w)) if w > h else (int(w * size / h), size)
    o = io.BytesIO()
    i.resize(s, Image.LANCZOS).save(o, format='PNG')
    o.seek(0)
    return o

def get_civdom(url: str) -> str | None:
    try:
        h = urlparse(url).netloc.lower()
        for d in CIVITAI:
            if d in h:
                return d
    except:
        pass
    return None

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

def get_fn(url):
    if any(x in url for x in [*CIVITAI, 'drive.google.com']): return None
    return Path(urlparse(url).path).name

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
    """
    Resolve a user-provided URL into a direct download URL when possible.
    Important fix: do NOT append ?token=... to CivitAI/Backblaze signed URLs (they are sensitive to modification).
    Only append TOKET for non-Civitai hosts when TOKET is set and needed.
    """

    civitai = get_civdom(url)

    def maybe_add_token(u):
        # Add token only for non-Civitai hosts and when TOKET is set.
        try:
            parsed = urlparse(u)
            host = parsed.netloc.lower()
        except:
            return u

        # If host is Civitai or Backblaze storage, do NOT modify the signed URL.
        if any(d in host for d in CIVITAI) or host.startswith('b2.'):
            return u

        if not TOKET:
            return u

        if '?type=' in u:
            return u.replace('?type=', f'?token={TOKET}&type=')
        return f'{u}?token={TOKET}'

    if 'github.com' in url:
        url = url.replace('/blob/', '/raw/')
        return maybe_add_token(url), None, None

    elif 'huggingface.co' in url:
        url = url.split('?')[0]
        h = {'User-Agent': 'Mozilla/5.0', **({'Authorization': f'Bearer {TOBRUT}'} if TOBRUT else {})}
        ext = ['.safetensors', '.pt', '.pth']
        j, versionId = None, None

        if fn and Path(fn).suffix.lower() in ext:
            try:
                res = requests.get(re.sub(r'/(resolve|blob)/', '/raw/', url), headers=h)
                t = re.search(r'oid sha256:([a-fA-F0-9]{64})', res.text)

                if t:
                    sha256 = t.group(1)
                    j = None

                    for c in CIVITAI:
                        try:
                            api_url = f'https://{c}/api/v1/model-versions/by-hash/{sha256}'
                            j_try = get_json(api_url, civitai_headers())

                            if not j_try:
                                continue

                            r = next(
                                (f for f in j_try.get('files', [])
                                if f.get('hashes', {}).get('SHA256', '').lower() == sha256.lower()),
                                None
                            )

                            if r:
                                j = j_try
                                break

                        except Exception:
                            continue

            except Exception:
                j = None

        url = url.replace('/blob/', '/resolve/')
        return maybe_add_token(url), j, versionId

    elif civitai in url:
        input_url = url
        url = url.split('?token=')[0] if '?token=' in url else url

        if f'{civitai}/api/download/models/' in url:
            versionId = url.split('models/')[1].split('/')[0].split('?')[0]
            api_url = f'https://{civitai}/api/v1/model-versions/{versionId}'
            j = get_json(api_url, civitai_headers())

            if j:
                v = get_civitai(j, versionId)
                if v:
                    return url, j, versionId

            return url, None, None

        elif f'{civitai}/models/' in url:
            versionId = None
            modelId = url.split('models/')[1].split('/')[0].split('?')[0]
            if '?modelVersionId=' in url:
                versionId = url.split('?modelVersionId=')[1]

            api_url = f'https://{civitai}/api/v1/models/{modelId}'
            j = get_json(api_url, civitai_headers())

            if not j or civitai_earlyAccess(j, versionId, civitai):
                return None, None, None

            v = get_civitai(j, versionId)
            if not v:
                print(f'Unable to find download URL for\n-> {input_url}\n')
                return None, None, None

            url = next((f.get('downloadUrl') for f in v.get('files', []) if f.get('downloadUrl')), None)
            if not url:
                print(f'Unable to find download URL for\n-> {input_url}\n')
                return None, None, None

            return url, j, versionId

    return maybe_add_token(url), None, None

def ariari(url, fp, fn):
    url, j, versionId = get_url(url, fn)
    if not url: return

    civitai = get_civdom(url)
    civitai_api = (f'{civitai}/api/download/models/' in url and bool(TOKET))

    if civitai_api:
        try:
            headers = {'User-Agent': civitai_headers()['User-Agent'], 'Authorization': f'Bearer {TOKET}'}
            request_url = url
            resp = requests.get(request_url, headers=headers, allow_redirects=True, stream=True, timeout=30)
            final_url = resp.url
            resp.close()

            if final_url and final_url != request_url: url = final_url
            else: print("  No redirect detected; aria2 will use Authorization header.")

        except Exception as e:
            print(f"  Preflight failed: {e}")
            print("  Falling back to aria2 with Authorization header.")

    cmd = [
        'aria2c',
        f"--header=User-Agent: {civitai_headers()['User-Agent'] if f'{civitai}' in url else 'Mozilla/5.0'}",
        '--allow-overwrite=true', '--console-log-level=error', '--stderr=true',
        '-c', '-x16', '-s16', '-k1M', '-j5'
    ]

    if f'{civitai}/api/download/models/' in url and TOKET: cmd.append(f"--header=Authorization: Bearer {TOKET}")
    if TOBRUT and 'huggingface.co' in url: cmd.append(f'--header=Authorization: Bearer {TOBRUT}')

    if fn: cmd += ['-o', fn]

    cmd.append(url)

    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        aria2_output, break_line, error_code, error_line = '', False, [], []

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

                    if re.match(r'\[#\w{6}\s.*\]', prog):
                        prog = re.sub(r'\[', MAGENTA + '【' + RESET, prog)
                        prog = re.sub(r'\]', MAGENTA + '】' + RESET, prog)
                        prog = re.sub(r'(#)(\w+)', f'{CYAN}\\1{RESET}{GREEN}\\2{RESET}', prog)
                        prog = re.sub(r'(\d+(\.\d+)?)(\w+)(/)(\d+(\.\d+)?)(\w+)', f"\\1{PURPLE}\\3{RESET}{MAGENTA}\\4{RESET}\\5{PURPLE}\\7{RESET}", prog)
                        prog = re.sub(r'(\()(\d+%)(\))', f'{MAGENTA}\\1{RESET}\\2{MAGENTA}\\3{RESET}', prog)
                        prog = re.sub(r'(CN)(:)(\d+)', f"{CYAN}\\1{RESET}\\2{ORANGE}\\3{RESET}", prog)
                        prog = re.sub(r'(DL)(:)(\d+(\.\d+)?)(\w+)', f"{CYAN}\\1{RESET}\\2\\3{PURPLE}\\5{RESET}", prog)
                        prog = re.sub(r'(ETA)(:)(\d+\w+)', f"{CYAN}\\1{RESET}\\2{YELLOW}\\3{RESET}", prog)

                        lines = prog.splitlines()
                        for line in lines:
                            print(f"\r{' '*300}\r {line}", end='')
                            sys.stdout.flush()

                        break_line = True
                        break

        civitai = None
        error = error_code + error_line
        for lines in error: print(f'  {lines}')

        break_line and print()

        stripe = aria2_output.find('======+====+===========')
        if stripe != -1:
            for lines in aria2_output[stripe:].splitlines():
                if '|' in lines and 'OK' in lines:
                    lines = re.sub(r'(\|\s*)(OK)(\s*\|)', f'\\1{GREEN}\\2{RESET}\\3', lines)
                    first, _, last = lines.rpartition('|')
                    last = re.sub(r'/', f'{ORANGE}/{RESET}', last)
                    lines = f'{first}|{last}'
                    print(f'  {lines}')

        if j:
            civitai_infotags(j, fp, fn, versionId)
            civitai_preview(j, fp, fn, versionId)

        p.wait()

    except KeyboardInterrupt:
        print(f'\n{"":>2}^ Canceled')

def curlly(cmd, fn):
    try:
        p = subprocess.Popen(
            shlex.split(cmd), cwd=str(Path.cwd()),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )

        prog = re.compile(r'(\d+\.\d+)%')
        curl_output = ''

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

def gdrown(url, fp=None, fn=None):
    is_folder = 'drive.google.com/drive/folders' in url
    cmd = f'gdown --fuzzy {url}'

    if fp:
        fp = Path(fp).expanduser()
        fp.mkdir(parents=True, exist_ok=True)
        if fn:
            fn = fp / fn
            cmd += f' -O {fn}'
        cwd = str(fp)
    else:
        cwd = None

    if fn and not fp: cmd += f' -O {fn}'
    if is_folder: cmd += ' --folder'

    SyS(f'cd {cwd} && {cmd}' if cwd else cmd)

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
    print()

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

def parse_size(size_str):
    match = re.match(r'^([\d\.]+)\s*([a-zA-Z]*)$', size_str.strip())
    if not match:
        return 0.0
    val, unit = match.groups()
    val = float(val)
    unit = unit.lower()
    if 'g' in unit:
        return val * 1024 * 1024 * 1024
    elif 'm' in unit:
        return val * 1024 * 1024
    elif 'k' in unit:
        return val * 1024
    return val

def format_size(bytes_val):
    if bytes_val >= 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024 * 1024):.1f}GiB"
    elif bytes_val >= 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f}MiB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f}KiB"
    return f"{bytes_val:.0f}B"

def is_git_repo(url):
    url_lower = url.lower().strip()
    if url_lower.endswith('.git'):
        return True
    if 'github.com/' in url_lower and not any(x in url_lower for x in ['/resolve/', '/raw/', '/releases/download/', '/archive/']):
        return True
    return False

def parallel_download(download_list, max_workers=3, parallel=True):
    tasks = []
    for idx, item in enumerate(download_list):
        url = item[0].strip()
        if not url:
            continue
        dest_dir = Path(item[1]).expanduser()
        filename = item[2] if len(item) > 2 else None
        
        if not filename:
            parts = url.split()
            if len(parts) > 1:
                url = parts[0]
                filename = parts[1]
                
        tasks.append({
            'index': idx + 1,
            'url': url,
            'dest_dir': dest_dir,
            'filename': filename,
            'status': 'Pending',
            'progress_line': '',
            'bytes_downloaded': 0.0,
            'bytes_total': 0.0,
            'speed_bytes': 0.0,
            'eta_secs': 0.0
        })
        
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No URLs to download.")
        return

    lock = Lock()
    completed_lines = []
    
    def update_display():
        with lock:
            clear_output(wait=True)
            for line in completed_lines:
                print(line)
            
            active_list = [t for t in tasks if t['status'] == 'Downloading']
            for t in active_list:
                if t['progress_line']:
                    print(t['progress_line'])
            
            if active_list:
                sum_downloaded = sum(t['bytes_downloaded'] for t in active_list)
                sum_total = sum(t['bytes_total'] for t in active_list)
                sum_speed = sum(t['speed_bytes'] for t in active_list)
                
                percentage = int((sum_downloaded / sum_total) * 100) if sum_total > 0 else 0
                eta_secs = int((sum_total - sum_downloaded) / sum_speed) if sum_speed > 0 else 0
                
                if eta_secs > 60:
                    eta_str = f"{eta_secs // 60}m{eta_secs % 60}s"
                else:
                    eta_str = f"{eta_secs}s"
                    
                downloaded_str = format_size(sum_downloaded)
                total_str = format_size(sum_total)
                speed_str = f"{format_size(sum_speed)}/s"
                
                colored_summary = f"\033[38;5;135m【\033[0m\033[36m#Parallel\033[0m \033[38;5;35m{downloaded_str}\033[0m/\033[38;5;135m{total_str}\033[0m(\033[35m{percentage}%\033[0m) DL:\033[38;5;69m{speed_str}\033[0m ETA:\033[33m{eta_str}\033[0m\033[38;5;135m】\033[0m"
                print(colored_summary)
            sys.stdout.flush()

    def run_single_download(task):
        url = task['url']
        dest_dir = task['dest_dir']
        filename = task['filename']
        task_idx = task['index']
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        cwd = Path.cwd()
        
        resolved_url, j, versionId = get_url(url, filename)
        if not resolved_url:
            with lock:
                task['status'] = 'Failed'
                completed_lines.append(f"  [{task_idx}/{total_tasks}] ❌ Failed to resolve URL: {url}")
            return
            
        civitai_dom = get_civdom(resolved_url)
        is_civitai = bool(civitai_dom)
        is_gdrive = 'drive.google.com' in resolved_url
        is_git = is_git_repo(resolved_url)
        
        if not filename:
            if is_civitai:
                filename = None
            else:
                filename = Path(urlparse(resolved_url).path).name
                if not filename or filename in ['resolve', 'raw']:
                    filename = None
        
        identifier = versionId or filename or Path(urlparse(resolved_url).path).name or "model"
        
        try:
            with lock:
                task['status'] = 'Downloading'
            
            if is_gdrive:
                cmd = f"gdown --fuzzy {resolved_url}"
                if filename:
                    cmd += f" -O {filename}"
                cmd += " --folder" if 'drive/folders' in resolved_url else ""
                
                p = subprocess.Popen(shlex.split(cmd), cwd=str(dest_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                while p.poll() is None:
                    time.sleep(0.5)
                p.wait()
                
            elif is_git:
                cmd = f"git clone {resolved_url}"
                if filename:
                    cmd += f" {filename}"
                p = subprocess.Popen(shlex.split(cmd), cwd=str(dest_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                while p.poll() is None:
                    with lock:
                        task['progress_line'] = f"\033[36m【Cloning {resolved_url}...】\033[0m"
                    update_display()
                    time.sleep(0.5)
                p.wait()
                
            else:
                cmd = [
                    'aria2c',
                    f"--header=User-Agent: {civitai_headers()['User-Agent'] if is_civitai else 'Mozilla/5.0'}",
                    '--allow-overwrite=true', '--console-log-level=error', '--stderr=true',
                    '-c', '-x16', '-s16', '-k1M', '-j5'
                ]
                
                if is_civitai and TOKET:
                    cmd.append(f"--header=Authorization: Bearer {TOKET}")
                if TOBRUT and 'huggingface.co' in resolved_url:
                    cmd.append(f'--header=Authorization: Bearer {TOBRUT}')
                    
                if filename:
                    cmd += ['-o', filename]
                cmd += ['--dir', str(dest_dir)]
                cmd.append(resolved_url)
                
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                while True:
                    line = p.stderr.readline()
                    if line == '' and p.poll() is not None:
                        break
                    if line:
                        for prog in line.splitlines():
                            if re.match(r'\[#\w{6}\s.*\]', prog):
                                parsed = parse_aria_line(prog)
                                
                                colored_prog = prog
                                colored_prog = re.sub(r'\[', f'\033[38;5;135m【\033[0m', colored_prog)
                                colored_prog = re.sub(r'\]', f'\033[38;5;135m】\033[0m', colored_prog)
                                colored_prog = re.sub(r'(#)(\w+)', f'\033[36m\\1\033[0m\033[32m\\2\033[0m', colored_prog)
                                colored_prog = re.sub(r'(\d+(\.\d+)?)(\w+)(/)(\d+(\.\d+)?)(\w+)', f"\\1\033[38;5;135m\\3\033[0m\033[35m\\4\033[0m\\5\033[38;5;135m\\7\033[0m", colored_prog)
                                colored_prog = re.sub(r'(\()(\d+%)(\))', f'\033[35m\\1\033[0m\\2\033[35m\\3\033[0m', colored_prog)
                                colored_prog = re.sub(r'(CN)(:)(\d+)', f"\033[36m\\1\033[0m:\033[38;5;208m\\3\033[0m", colored_prog)
                                colored_prog = re.sub(r'(DL)(:)(\d+(\.\d+)?)(\w+)', f"\033[36m\\1\033[0m:\\3\033[38;5;135m\\5\033[0m", colored_prog)
                                colored_prog = re.sub(r'(ETA)(:)(\d+\w+)', f"\033[36m\\1\033[0m:\033[33m\\3\033[0m", colored_prog)
                                
                                with lock:
                                    task['progress_line'] = colored_prog
                                    if parsed:
                                        task['bytes_downloaded'] = parsed['downloaded']
                                        task['bytes_total'] = parsed['total']
                                        task['speed_bytes'] = parsed['speed']
                                        task['eta_secs'] = parsed['eta']
                                update_display()
                p.wait()
                
                if is_civitai and j:
                    try:
                        actual_fn = filename
                        if not actual_fn:
                            v_info = get_civitai(j, versionId)
                            if v_info:
                                actual_fn = v_info.get('files', [{}])[0].get('name')
                        if actual_fn:
                            civitai_infotags(j, dest_dir, actual_fn, versionId)
                            civitai_preview(j, dest_dir, actual_fn, versionId)
                    except:
                        pass
            with lock:
                task['status'] = 'Completed'
                completed_lines.append(f"  [\033[32m{task_idx}\033[0m/\033[32m{total_tasks}\033[0m] \033[32m✓\033[0m {identifier}")
            update_display()
        except Exception as e:
            with lock:
                task['status'] = 'Failed'
                completed_lines.append(f"  [{task_idx}/{total_tasks}] ❌ Failed: {e}")
            update_display()

    def parse_aria_line(line):
        try:
            match = re.search(r'\[#\w{6}\s+([\d\.]+)([a-zA-Z]+)/([\d\.]+)([a-zA-Z]+)\((\d+)%\).*?DL:([\d\.]+)([a-zA-Z]+)(?:\s+ETA:(\w+))?\]', line)
            if not match:
                return None
            dl_val, dl_unit, tot_val, tot_unit, pct, sp_val, sp_unit, eta_str = match.groups()
            downloaded = parse_size(f"{dl_val}{dl_unit}")
            total = parse_size(f"{tot_val}{tot_unit}")
            speed = parse_size(f"{sp_val}{sp_unit}")
            eta = 0
            if eta_str:
                eta_match = re.match(r'(?:(\d+)m)?(?:(\d+)s)?', eta_str)
                if eta_match:
                    m, s = eta_match.groups()
                    eta = (int(m or 0) * 60) + int(s or 0)
            return {'downloaded': downloaded, 'total': total, 'speed': speed, 'eta': eta}
        except:
            return None

    if parallel and max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(run_single_download, tasks)
    else:
        for task in tasks:
            run_single_download(task)
