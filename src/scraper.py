import os
import re
import json
import time
import requests
import tempfile
import tarfile
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\s]", re.VERBOSE)
_LATINIZIZER = str.maketrans({
    "ß": "ss", "ẞ": "ss",
    "æ": "ae", "Æ": "ae",
    "œ": "oe", "Œ": "oe",
    "ø": "o",  "Ø": "o",
    "đ": "d",  "Đ": "d",
    "ł": "l",  "Ł": "l",
    "þ": "th", "Þ": "th",
    "ı": "i",
})

def latinize(text):
    text = text.translate(_LATINIZIZER)
    text = unicodedata.normalize("NFKD", text)
    combining = unicodedata.combining
    text = "".join(ch for ch in text if not combining(ch))
    return text.encode("ascii", "ignore").decode("ascii")

def skip_ws(s, i):
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    return i

def read_balanced(s, i, open_ch, close_ch):
    n = len(s)
    i = skip_ws(s, i)
    if i >= n or s[i] != open_ch:
        return None, None
    depth = 1
    j = i + 1
    while j < n and depth:
        ch = s[j]
        if ch == "\\":
            j += 2
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
        j += 1
    if depth != 0:
        return None, None
    return s[i + 1 : j - 1], j

def skip_balanced(s, i, open_ch, close_ch):
    n = len(s)
    i = skip_ws(s, i)
    if i >= n or s[i] != open_ch:
        return None
    depth = 1
    i += 1
    while i < n and depth:
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
        i += 1
    return i if depth == 0 else None

def scrape_arxiv_sources(
    start,
    end,
    category = "hep-th",
    include_crosslists = True,
    request_delay = 3,
    max_retries = 3,
    timeout = 30,
    data_dir = "../data",
    overwrite = False
):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError("The end date must not be earlier than the start date.")
    start_str = start_dt.strftime("%Y%m%d0000")
    end_str = end_dt.strftime("%Y%m%d2359")
    search_query = f"cat:{category} AND submittedDate:[{start_str} TO {end_str}]"
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok = True)
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    processed_ids = set()
    if not overwrite and os.path.isdir(raw_dir):
        for name in os.listdir(raw_dir):
            if os.path.isdir(os.path.join(raw_dir, name)):
                processed_ids.add(name)
    api_url = "http://export.arxiv.org/api/query"
    per_page = 100
    start_index = 0
    processed_count = 0
    downloaded_count = 0
    failed_count = 0
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    while True:
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
            "start": start_index,
            "max_results": per_page,
        }
        resp = requests.get(api_url, params = params, timeout = timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ns)
        if not entries:
            break
        for entry in entries:
            id_url = entry.findtext("atom:id", namespaces = ns)
            arxiv_id_version = id_url.rsplit("/", 1)[-1]
            base_id = arxiv_id_version.split("v", 1)[0]
            categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", ns)]
            primary_category = categories[0] if categories else ""
            if not include_crosslists and primary_category != category:
                continue
            if not overwrite and base_id in processed_ids:
                continue
            processed_count += 1
            paper_dir = os.path.join(raw_dir, base_id)
            source_path = os.path.join(paper_dir, "source.tar.gz")
            os.makedirs(paper_dir, exist_ok = True)
            ok = False
            for _ in range(max_retries):
                try:
                    url = f"https://arxiv.org/e-print/{base_id}"
                    res = requests.get(url, stream = True, timeout = timeout)
                    if res.status_code == 200:
                        with open(source_path, "wb") as f:
                            for chunk in res.iter_content(chunk_size = 8192):
                                if chunk:
                                    f.write(chunk)
                        processed_ids.add(base_id)
                        downloaded_count += 1
                        ok = True
                        title = entry.findtext("atom:title", namespaces = ns).strip()
                        abstract = entry.findtext("atom:summary", namespaces = ns).strip()
                        authors = [
                            a.findtext("atom:name", namespaces = ns).strip()
                            for a in entry.findall("atom:author", ns)
                        ]
                        submitted = entry.findtext("atom:published", namespaces = ns).strip()
                        updated = entry.findtext("atom:updated", namespaces = ns).strip()
                        record = {
                            "arxiv_id": base_id,
                            "title": title,
                            "authors": authors,
                            "abstract": abstract,
                            "categories": categories,
                            "dates": {
                                "submitted": submitted,
                                "updated": updated,
                            },
                        }
                        with open(metadata_path, "a", encoding = "utf-8") as md_file:
                            md_file.write(json.dumps(record, ensure_ascii = False) + "\n")
                        time.sleep(request_delay)
                        break
                except Exception:
                    pass
                time.sleep(request_delay)
            if not ok:
                failed_count += 1
        start_index += per_page
        time.sleep(request_delay)

    total = processed_count
    successful = downloaded_count
    failed = failed_count
    if total > 0:
        p_success = round(100 * successful / total, 2)
        p_failed = round(100 * failed / total, 2)
    else:
        p_success = 0
        p_failed = 0

    print(f"\ntotal papers between {start} and {end}:", total)
    print(f"successful retrievals: {successful} ({p_success} %)")
    print(f"failed retrievals: {failed} ({p_failed} %)")

def load_metadata(metadata_path = "../data/metadata.jsonl"):
    metadata = {}
    if not os.path.exists(metadata_path):
        return metadata
    with open(metadata_path, "r", encoding = "utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            arxiv_id = record.get("arxiv_id")
            if not arxiv_id:
                continue
            metadata[arxiv_id] = record
    return metadata

def extract_macros(latex, package = False):
    if package:
        tex = re.sub(r"(?<!\\)%.*", "", latex)
        must_re = re.compile(r"(\\renewcommand\s*\{?\s*\\maketitle\b)|(\\@startsection\b|\\renewcommand\s*\{?\s*\\(?:sub)*section\b)", flags = re.IGNORECASE | re.DOTALL)
        marker_patterns = [r"\\PackageWarningNoLine\b", r"\\ps@\w+", r"\\pagestyle\b|\\thispagestyle\b|\\pagenumbering\b", r"\\textwidth\b|\\textheight\b|\\oddsidemargin\b|\\evensidemargin\b|\\topmargin\b|\\headheight\b|\\headsep\b|\\footskip\b|\\voffset\b|\\hoffset\b", r"\\titleformat\b|\\RequirePackage\s*(?:\[[^\]]*\]\s*)?\{\s*titlesec\s*\}"]
        marker_re = re.compile("|".join(marker_patterns), flags = re.IGNORECASE | re.DOTALL)
        if must_re.search(tex) and marker_re.search(tex):
            return {"noarg_mac": {}, "arg_mac": {}, "env_mac": {}, "delim_mac": {}}

    _GUARD_NAMES = {"chapter", "section", "maketitle"}
    noarg_mac = {}
    arg_mac = {}
    env_mac = {}
    delim_mac = {}

    def read_cmd_name(s, i):
        i = skip_ws(s, i)
        if i >= len(s) or s[i] != "\\":
            return None, None
        i += 1
        i = skip_ws(s, i)
        j = i
        while j < len(s) and (s[j].isalpha() or s[j] == "@"):
            j += 1
        if j == i:
            return None, None
        return s[i:j], j

    def read_newcommand_name(s, i):
        i = skip_ws(s, i)
        if i < len(s) and s[i] == "{":
            inner, j = read_balanced(s, i, "{", "}")
            if inner is None:
                return None, None
            inner = inner.strip()
            if not inner.startswith("\\"):
                return None, None
            nm, _ = read_cmd_name(inner, 0)
            return nm, j
        return read_cmd_name(s, i)

    def set_noarg(name, body):
        if name in _GUARD_NAMES:
            return
        if name and name not in arg_mac and name not in delim_mac and name not in noarg_mac:
            noarg_mac[name] = body

    def set_arg(name, n_args):
        if name in _GUARD_NAMES:
            return
        if name and name not in noarg_mac and name not in delim_mac and name not in arg_mac and n_args > 0:
            arg_mac[name] = n_args

    def set_delim(name, endname):
        if name in _GUARD_NAMES or endname in _GUARD_NAMES:
            return
        if name and endname and name not in noarg_mac and name not in arg_mac and name not in delim_mac:
            delim_mac[name] = endname

    # env_mac: \newenvironment
    newenv_kw = re.compile(r"\\newenvironment\b", flags=re.DOTALL)
    pos = 0
    while True:
        m = newenv_kw.search(latex, pos)
        if not m:
            break
        i = skip_ws(latex, m.end())
        if i < len(latex) and latex[i] == "*":
            i += 1
        name, j = read_balanced(latex, i, "{", "}")
        if name is None:
            pos = m.end()
            continue
        begin_body, j2 = read_balanced(latex, j, "{", "}")
        if begin_body is None:
            pos = m.end()
            continue
        end_body, j3 = read_balanced(latex, j2, "{", "}")
        if end_body is None:
            pos = m.end()
            continue
        nm = (name or "").strip()
        if nm and nm not in env_mac:
            mb = re.search(r"\\\s*begin\s*\{\s*([^}]+)\s*\}", begin_body)
            env_begin = (mb.group(1) or "").strip() if mb else ""
            me = None
            for me2 in re.finditer(r"\\\s*end\s*\{\s*([^}]+)\s*\}", end_body):
                me = me2
            env_end = (me.group(1) or "").strip() if me else ""
            if env_begin and env_begin == env_end:
                env_mac[nm] = env_begin
        pos = j3

    # \def... (including delimited \def\bal#1\eal{...})
    def_kw = re.compile(r"\\(?:def|gdef|edef|xdef)\b", flags = re.DOTALL)
    pos = 0
    while True:
        m = def_kw.search(latex, pos)
        if not m:
            break
        name, j = read_cmd_name(latex, m.end())
        if not name:
            pos = m.end()
            continue
        k = j
        n = len(latex)
        while k < n and latex[k] != "{":
            k += 1
        if k >= n:
            pos = m.end()
            continue
        param_txt = latex[j:k]
        n_args = len(re.findall(r"#\s*\d+", param_txt))
        endname = None
        if n_args == 1 and re.search(r"#\s*1\b", param_txt):
            t = re.search(r"#\s*1\b", param_txt).end()
            t = skip_ws(param_txt, t)
            if t < len(param_txt) and param_txt[t] == "\\":
                endname, _ = read_cmd_name(param_txt, t)
        body, k2 = read_balanced(latex, k, "{", "}")
        if body is None:
            pos = m.end()
            continue
        if endname:
            set_delim(name, endname)
        elif n_args > 0:
            set_arg(name, n_args)
        else:
            set_noarg(name, body.strip())
        pos = k2

    # \newcommand / \renewcommand / \providecommand / \DeclareRobustCommand
    cmd_kw = re.compile(r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\b", flags = re.DOTALL)
    pos = 0
    while True:
        m = cmd_kw.search(latex, pos)
        if not m:
            break
        i = skip_ws(latex, m.end())
        if i < len(latex) and latex[i] == "*":
            i += 1
        name, j = read_newcommand_name(latex, i)
        if not name:
            pos = m.end()
            continue
        n_args = 0
        j = skip_ws(latex, j)
        if j < len(latex) and latex[j] == "[":
            inside, j2 = read_balanced(latex, j, "[", "]")
            if inside is None:
                pos = m.end()
                continue
            mm = re.match(r"\s*(\d+)\s*\Z", inside)
            if mm:
                n_args = int(mm.group(1))
                j = j2
        j = skip_ws(latex, j)
        if n_args and j < len(latex) and latex[j] == "[":
            _, j2 = read_balanced(latex, j, "[", "]")
            if j2 is None:
                pos = m.end()
                continue
            j = j2
        body, j2 = read_balanced(latex, j, "{", "}")
        if body is None:
            pos = m.end()
            continue
        if n_args > 0:
            set_arg(name, n_args)
        else:
            set_noarg(name, body.strip())
        pos = j2

    return {"noarg_mac": noarg_mac, "arg_mac": arg_mac, "env_mac": env_mac, "delim_mac": delim_mac}

def replace_macros(latex, macros):
    noarg_mac = (macros or {}).get("noarg_mac", {}) or {}
    arg_mac = (macros or {}).get("arg_mac", {}) or {}
    env_mac = (macros or {}).get("env_mac", {}) or {}
    delim_mac = (macros or {}).get("delim_mac", {}) or {}

    def strip_macro_definitions(s):
        n = len(s)
        out = []
        i = 0

        def read_cmd_name_at(s2, p):
            p = skip_ws(s2, p)
            if p >= len(s2) or s2[p] != "\\":
                return None, None
            p += 1
            p = skip_ws(s2, p)
            if p >= len(s2):
                return None, None
            c = s2[p]
            if c.isalpha() or c == "@":
                q = p
                while q < len(s2) and (s2[q].isalpha() or s2[q] == "@"):
                    q += 1
                return s2[p:q], q
            return s2[p : p + 1], p + 1

        while i < n:
            k = s.find("\\", i)
            if k == -1:
                out.append(s[i:])
                break
            out.append(s[i:k])

            # \def...
            if s.startswith(r"\def", k):
                p = k + 4
                name, p2 = read_cmd_name_at(s, p)
                if not name:
                    out.append(s[k:k+1]); i = k + 1; continue
                p = p2
                while p < n and s[p] != "{":
                    p += 1
                if p >= n:
                    out.append(s[k:k+1]); i = k + 1; continue
                end = skip_balanced(s, p, "{", "}")
                if end is None:
                    out.append(s[k:k+1]); i = k + 1; continue
                out.append(" ")
                i = end
                continue

            # \newcommand / \renewcommand / \providecommand
            matched = False
            for cmd in (r"\newcommand", r"\renewcommand", r"\providecommand"):
                if not s.startswith(cmd, k):
                    continue
                p = skip_ws(s, k + len(cmd))
                if p < n and s[p] == "*":
                    p += 1
                p = skip_ws(s, p)
                if p < n and s[p] == "{":
                    end_name = skip_balanced(s, p, "{", "}")
                    if end_name is None:
                        break
                    p = end_name
                else:
                    name, p2 = read_cmd_name_at(s, p)
                    if not name:
                        break
                    p = p2
                p = skip_ws(s, p)
                if p < n and s[p] == "[":
                    end = skip_balanced(s, p, "[", "]")
                    if end is None:
                        break
                    p = skip_ws(s, end)
                if p < n and s[p] == "[":
                    end = skip_balanced(s, p, "[", "]")
                    if end is None:
                        break
                    p = skip_ws(s, end)
                p = skip_ws(s, p)
                if p >= n or s[p] != "{":
                    break
                end_body = skip_balanced(s, p, "{", "}")
                if end_body is None:
                    break
                out.append(" ")
                i = end_body
                matched = True
                break
            if matched:
                continue
            out.append(s[k:k+1])
            i = k + 1

        return "".join(out)

    # 1) macros definitions
    latex = strip_macro_definitions(latex)

    # 2) noarg macros
    for name, body in sorted(noarg_mac.items(), key = lambda x: len(x[0]), reverse = True):
        latex = re.sub(r"\\\s*" + re.escape(name) + r"\b", lambda _m, body = body: body, latex)

    # 3) env aliases
    for name, env in sorted(env_mac.items(), key = lambda x: len(x[0]), reverse = True):
        latex = re.sub(r"\\begin\s*\{\s*" + re.escape(name) + r"\s*\}", r"\\begin{" + env + r"}", latex)
        latex = re.sub(r"\\end\s*\{\s*" + re.escape(name) + r"\s*\}", r"\\end{" + env + r"}", latex)

    # 4) delimited macros: \start ... \end -> @@env@@
    if delim_mac:
        names = sorted(delim_mac.keys(), key = len, reverse = True)
        start_re = re.compile(r"\\\s*(" + "|".join(re.escape(nm) for nm in names) + r")\b")
        out = []
        i = 0
        n = len(latex)
        while i < n:
            m = start_re.search(latex, i)
            if not m:
                out.append(latex[i:])
                break
            out.append(latex[i:m.start()])
            start = m.group(1)
            j = m.end()
            endname = delim_mac.get(start)
            if not endname:
                out.append(latex[m.start():m.end()])
                i = m.end()
                continue
            end_re = re.compile(r"\\\s*" + re.escape(endname) + r"\b")
            me = end_re.search(latex, j)
            if not me:
                out.append(latex[m.start():m.end()])
                i = m.end()
                continue
            out.append(" @@env@@ ")
            i = me.end()
        latex = "".join(out)

    # 5) macros w/ arguments -> @@env@@
    if arg_mac:
        names = sorted(arg_mac.keys(), key = len, reverse = True)
        cmd_re = re.compile(r"\\\s*(" + "|".join(re.escape(n) for n in names) + r")\b")
        out = []
        i = 0
        n = len(latex)
        while i < n:
            m = cmd_re.search(latex, i)
            if not m:
                out.append(latex[i:])
                break
            out.append(latex[i:m.start()])
            name = m.group(1)
            j = m.end()
            j = skip_ws(latex, j)
            if j < n and latex[j] == "[":
                k = skip_balanced(latex, j, "[", "]")
                if k is None:
                    out.append(latex[m.start():m.end()])
                    i = m.end()
                    continue
                j = k
            ok = True
            for _ in range(arg_mac[name]):
                j = skip_ws(latex, j)
                k = skip_balanced(latex, j, "{", "}")
                if k is None:
                    ok = False
                    break
                j = k
            if ok:
                out.append(" @@env@@ ")
                i = j
            else:
                out.append(latex[m.start():m.end()])
                i = m.end()
        latex = "".join(out)

    return latex

def merge_macros(dst, src):
    if not dst:
        return src
    if not src:
        return dst
    for k in ("noarg_mac", "arg_mac", "env_mac", "delim_mac"):
        d = dst.get(k)
        s = src.get(k)
        if d is None:
            dst[k] = dict(s) if s else {}
            continue
        if not s:
            continue
        for name, val in s.items():
            if name not in d:
                d[name] = val
    return dst

def find_custom_sty_packages(latex, root_dir):
    tex = re.sub(r'(?<!\\)%.*', '', latex)
    pkg_names = set()
    use_re = re.compile(r'\\usepackage\s*(?:\[[^\]]*\]\s*)?\{([^}]*)\}', flags = re.IGNORECASE)
    req_re = re.compile(r'\\RequirePackage\s*(?:\[[^\]]*\]\s*)?\{([^}]*)\}', flags = re.IGNORECASE)
    for m in use_re.finditer(tex):
        inner = m.group(1) or ""
        for nm in inner.split(","):
            nm = nm.strip()
            if nm:
                pkg_names.add(nm)
    for m in req_re.finditer(tex):
        inner = m.group(1) or ""
        for nm in inner.split(","):
            nm = nm.strip()
            if nm:
                pkg_names.add(nm)
    if not pkg_names:
        return []
    skip = {"jheppub", "jheppub-nosort", "jhep", "jcappub", "jinstpub", "pos", "revtex", "revtex4", "revtex4-1", "revtex4-2", "elsarticle", "svjour3", "svjour2", "llncs", "iopart", "mnras", "aipproc"}
    pkg_names = {(nm[:-4] if nm.lower().endswith(".sty") else nm) for nm in pkg_names if (nm[:-4] if nm.lower().endswith(".sty") else nm).lower() not in skip}
    if not pkg_names:
        return []
    sty_index = {}
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".sty"):
                base = fn[:-4]
                if base not in sty_index:
                    sty_index[base] = os.path.join(r, fn)
    custom_paths = []
    for nm in sorted(pkg_names):
        path = sty_index.get(nm)
        if path:
            custom_paths.append(path)
    return custom_paths

def clean_body(body):
    def strip_envs(text, env_pattern, placeholder):
        full_pat = r'\\begin\s*\{\s*(' + env_pattern + r')\s*\*?\s*\}\s*(?:\[[^\]]*\]\s*)?[\s\S]*?\\end\s*\{\s*\1\s*\*?\s*\}'
        open_pat = r'\\begin\s*\{\s*(' + env_pattern + r')\s*\*?\s*\}\s*(?:\[[^\]]*\]\s*)?[\s\S]*\Z'
        text = re.sub(full_pat, f' {placeholder} ', text)
        text = re.sub(open_pat, f' {placeholder} ', text)
        return text

    def strip_arg_cmds(s, repl):
        n = len(s)
        out = []
        i = 0
        cmds = sorted(repl.keys(), key = len, reverse = True)
        while i < n:
            k = s.find("\\", i)
            if k == -1:
                out.append(s[i:])
                break
            out.append(s[i:k])
            for cmd in cmds:
                if s.startswith(cmd, k):
                    j = skip_ws(s, k + len(cmd))
                    if j < n and s[j] == "*":
                        j += 1
                    j = skip_ws(s, j)
                    if j < n and s[j] == "[":
                        j2 = skip_balanced(s, j, "[", "]")
                        if j2 is None:
                            out.append(s[k:k+1])
                            i = k + 1
                            break
                        j = skip_ws(s, j2)
                    end = skip_balanced(s, j, "{", "}")
                    if end is None:
                        out.append(s[k:k+1])
                        i = k + 1
                    else:
                        out.append(repl[cmd])
                        i = end
                    break
            else:
                out.append(s[k:k+1])
                i = k + 1
        return "".join(out)

    def replace_inline_dollar_math(s):
        n = len(s)
        out = []
        i = 0
        while i < n:
            if s[i] != '$' or (i > 0 and s[i - 1] == '\\' and not (i > 1 and s[i - 2] == '\\')):
                out.append(s[i])
                i += 1
                continue
            j = i + 1
            depth = 0
            while j < n:
                cj = s[j]
                if cj == '\\':
                    j += 2
                    continue
                if cj == '{':
                    depth += 1
                elif cj == '}':
                    if depth > 0:
                        depth -= 1
                elif cj == '$' and depth == 0:
                    inner = s[i + 1 : j]
                    m_simple = re.match(r'\s*([A-Za-z]|\d+(?:\.\d+)?)\s*\Z', inner)
                    if m_simple:
                        out.append(' ' + m_simple.group(1) + ' ')
                    else:
                        m_dim = re.match(r'\s*(\d+)\s*([dD])\s*\Z', inner)
                        if m_dim:
                            out.append(' ' + m_dim.group(1) + m_dim.group(2) + ' ')
                        else:
                            STYLE = r'(?:mathcal|mathbb|mathbf|mathrm|mathit|mathsf|mathtt|mathfrak|mathscr|bf|it|rm|sf|tt|cal|frak|scr)'
                            SPACE = r'(?:\\(?:,|;|:|!|quad|qquad|enspace)\s*)*'
                            m_cmd_single = re.match(r'^\s*\{?\s*\\[A-Za-z@]+\s*(?:\{\s*' + SPACE + r'([A-Za-z])\s*' + SPACE + r'\}|\s+([A-Za-z]))\s*\}?\s*\Z', inner)
                            if m_cmd_single:
                                ch = m_cmd_single.group(1) or m_cmd_single.group(2)
                                out.append(' ' + ch + ' ')
                            else:
                                m_assign = re.match(r'^\s*\{?\s*(?:([A-Za-z])|\\' + STYLE + r'\s*(?:\{\s*([A-Za-z])\s*\}|([A-Za-z])))\s*\}?\s*=\s*([A-Za-z]+|-?\d+(?:\.\d+)?)\s*\Z', inner)
                                if m_assign:
                                    lhs = m_assign.group(1) or m_assign.group(2) or m_assign.group(3)
                                    rhs = m_assign.group(4)
                                    out.append(' ' + lhs + ' = ' + rhs + ' ')
                                else:
                                    out.append(' @@math@@ ')
                    i = j + 1
                    break
                j += 1
            else:
                out.append(s[i])
                i += 1
                continue
        return ''.join(out)
    
    def strip_unwanted_blocks(s):
        levels = {'section': 2, 'subsection': 3, 'subsubsection': 4, 'paragraph': 5, 'subparagraph': 6}
        head_re = re.compile(r'\\(?P<cmd>section|subsection|subsubsection|paragraph|subparagraph)\s*\*?\s*(?:\[[^\]]*\]\s*)?', flags = re.IGNORECASE)
        bad_re = re.compile(r'\b(?:ac?know[a-z]*ment(?:s)?|dedication(?:s)?|abstract(?:s)?|reference(?:s)?|bibliograph(?:y|ies)|bibliorgraph(?:y|ies)|appendix(?:es)?|appendices)\b', flags = re.IGNORECASE)
        heads = []
        pos = 0
        n = len(s)
        while True:
            m = head_re.search(s, pos)
            if not m:
                break
            title, j = read_balanced(s, m.end(), "{", "}")
            if title is None or j is None:
                pos = m.end()
                continue
            cmd = (m.group('cmd') or '').lower()
            lvl = levels.get(cmd, 99)
            heads.append((m.start(), j, lvl, title))
            pos = j
        if not heads:
            return s
        cuts = []
        for idx, (start, title_end, lvl, title) in enumerate(heads):
            if not bad_re.search(title or ''):
                continue
            end_cut = n
            for k in range(idx + 1, len(heads)):
                if heads[k][2] <= lvl:
                    end_cut = heads[k][0]
                    break
            cuts.append((start, end_cut))
        if not cuts:
            return s
        cuts.sort()
        merged = []
        for a, b in cuts:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)
        out = []
        last = 0
        for a, b in merged:
            out.append(s[last:a])
            last = b
        out.append(s[last:])
        return ''.join(out)
    
    def strip_tikzstyle_defs(s):
        out = []
        i = 0
        n = len(s)
        head_re = re.compile(r"\\tikzstyle\s*\{[^}]*\}", flags=re.DOTALL)
        while True:
            m = head_re.search(s, i)
            if not m:
                out.append(s[i:])
                break
            out.append(s[i:m.start()])
            p = m.end()
            p = skip_ws(s, p)
            if p + 1 < n and s[p] == "+" and s[p + 1] == "=":
                p += 2
            elif p < n and s[p] == "=":
                p += 1
            else:
                out.append(" ")
                i = p
                continue
            p = skip_ws(s, p)
            end = skip_balanced(s, p, "[", "]")
            if end is None:
                out.append(" ")
                i = p
                continue
            out.append(" ")
            i = end
        return "".join(out)

    # normalize LaTeX accent macros to base letters
    body = re.sub(r'\{\\["\'`^~.]([A-Za-z])\}', r'\1', body)
    body = re.sub(r'\\["\'`^~.]([A-Za-z])', r'\1', body)

    # front-matter noise
    body = re.sub(r'\\begin\s*{abstract}[\s\S]*?\\end\s*{abstract}', ' ', body, flags = re.IGNORECASE)
    body = re.sub(r'\\begin\s*{frontmatter}[\s\S]*?\\end\s*{frontmatter}', ' ', body, flags = re.IGNORECASE)
    body = re.sub(r'\\(?:title|author|email|affiliation|address|institute|date|keywords|pacs|subjclass|thanks)\s*\*?\s*(?:\[[^\]]*\]\s*)?\{(?:[^{}]|\{[^{}]*\})*\}', ' ', body)

    # inline math and code modes
    body = strip_arg_cmds(body, {r'\ensuremath': ' @@math@@ ', r'\lstinputlisting': ' @@code@@ ', r'\inputminted': ' @@code@@ ', r'\verbatiminput': ' @@code@@ '})
    body = re.sub(r'\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]', ' @@eqn@@ ', body)
    body = re.sub(r'\\\([\s\S]*?\\\)', ' @@math@@ ', body)
    body = replace_inline_dollar_math(body)

    # quotations and non-breaking spaces
    body = body.translate(str.maketrans('', '', "\"'`"))
    body = body.replace('~', ' ')

    # undesired environments
    body = strip_envs(body, r'(?:equation|eqnarray|align|alignat|gather|multline|flalign|flalignat|xalignat|xxalignat|displaymath|IEEEeqnarray|dmath|dgroup|CD|cases|numcases|subnumcases)', ' @@eqn@@ ')
    body = re.sub(r'\\begin\s*{empheq}\s*(?:\[[^\]]*\]\s*)?\{[^}]*\}[\s\S]*?\\end\s*{empheq}', ' @@eqn@@ ', body)
    body = strip_envs(body, r'(?:figure|table|tabular|tabularx|longtable|supertabular|wrapfigure|wraptable|sidewaysfigure|sidewaystable|subfigure|subtable|ytableau|tikzpicture|pgfpicture|pspicture|overpic|picture|axis|semilogxaxis|semilogyaxis|loglogaxis|groupplot|polaraxis|fmffile|fmfgraph|fmfsubgraph|feynman|feynmandiagram|diagram|tikzcd|xy|xymatrix)', ' @@fig@@ ')
    body = re.sub(r'\\(?P<blk>tikzpicture|pgfpicture|pspicture|overpic|xy|xymatrix|diagram)\s*(?:\[[^\]]*\]\s*)?[\s\S]*?\\end(?P=blk)\b', ' @@fig@@ ', body)
    body = strip_envs(body, r'(?:verbatim|Verbatim|BVerbatim|lstlisting|minted|alltt|comment|mmaCell)', ' @@code@@ ')
    body = strip_envs(body, r'(?:algorithm|algorithmic|algorithmicx|algorithm2e|pseudocode|pseudo)', ' @@alg@@ ')
    body = strip_envs(body, r'(?:python|py|ipython|julia|matlab|octave|fortran|c\\+\\+|cpp|java|rust|go|bash|sh)', ' @@code@@ ')
    body = re.sub(r'\\begin\s*{(?:minipage|multicols)}\s*\{[^}]*\}', ' ', body)

    # common custom macros
    body = re.sub(r'\\(?:be|beq|bea|ben)\b[\s\S]*?\\(?:ee|eeq|eea|een)\b', ' @@eqn@@ ', body)
    body = re.sub(r'\\makeatletter[\s\S]*?\\makeatother', ' ', body)

    # figures, labels/refs, citations, urls, list items
    body = re.sub(r'\\label(?:[A-Za-z@]+)?\s*{[^}]*}', '', body)
    body = re.sub(r'\\includegraphics\*?\s*(\[[^\]]*\]\s*)?\{(?:[^{}]|\{[^{}]*\})*\}', ' @@fig@@ ', body)
    body = re.sub(r'\\(?:ref|eqref|pageref|cref|Cref|cpageref|Cpageref|namecref|nameCref|autoref|hyperref|nameref|vref|Vref|vrefrange|Vrefrange|labelcref|labelcpageref)\s*(?:\[[^\]]*\]\s*)?\{[^}]*\}', ' @@ref@@ ', body)
    body = re.sub(r'\\(?:cite|citen|citep|citet|citealp|citealt|citeauthor|citeyear|citeyearpar|citeyearnp|citeyearalt|citeyearalp|nocite|parencite|textcite|footcite|autocite|supercite|smartcite|citepalias|citetalias|citepalias|citenum|citetext|citefield|citeurl|fullcite|footfullcite|cites|parencites|textcites|footcites|autocites|smartcites|supercites|Cite|Citep|Citet|Citealt|Parencite|Textcite|Footcite|Autocite|Smartcite|Cites|Parencites|Textcites|Footcites|Autocites|Smartcites|Supercites)\*?\s*(?:\[[^\]]*\]\s*)*(?:\{[^}]*\}\s*)+', ' @@cite@@ ', body)
    body = re.sub(r'\\(?:url\s*{[^}]*}|nolinkurl\s*{[^}]*}|href\s*{[^}]*}\s*{[^}]*}|hyperlink\s*{[^}]*}\s*{[^}]*}|hyperref\s*\[[^\]]*\]\s*(?:{[^}]*})?)', ' @@url@@ ', body)
    body = re.sub(r'\\item\s*\[[^\]]*\]', ' ', body)

    # toc/aux and hyperref bookmarks
    body = re.sub(r'\\(?:addcontentsline\s*\{[^}]*\}\s*\{[^}]*\}\s*\{[\s\S]*?\}|addtocontents\s*\{[^}]*\}\s*\{[\s\S]*?\}|pdfbookmark\s*(?:\[[^\]]*\]\s*)?\{[^}]*\}\s*\{[^}]*\}|currentpdfbookmark\s*\{[^}]*\}\s*\{[^}]*\}|subpdfbookmark\s*\{[^}]*\}\s*\{[^}]*\}|printbibliography\s*(?:\[[^\]]*\]\s*)?)', ' ', body)
    body = re.sub(r'\\phantomsection\b', ' ', body)
    body = re.sub(r'\\(?:addtocounter|setcounter)\s*\{[^}]*\}\s*\{[^}]*\}', ' ', body)

    # remaning subsection structure
    body = strip_unwanted_blocks(body)
    body = strip_arg_cmds(body, {r'\part': ' ', r'\chapter': ' ', r'\section': ' ', r'\subsection': ' ', r'\subsubsection': ' ', r'\paragraph': ' ', r'\subparagraph': ' '})

    # unwrap common commands/environments
    body = re.sub(r'\\footnote\s*\*?\s*(?:\[[^\]]*\]\s*)?\{([^{}]*)\}', r' \1 ', body)
    body = re.sub(r'\\(?:textit|textbf|emph|underline|texttt|textsc|textrm|textsf|textsl|textup|textmd|textnormal|mbox|fbox|phantom|hphantom|vphantom)\s*\*?\s*(?:\[[^\]]*\]\s*)?\{([^{}]*)\}', r' \1 ', body)
    body = re.sub(r'\\(?:verb|lstinline)(.)(.*?)\1|\\mintinline\s*\{[^}]*\}(.)(.*?)\2', ' @@code@@ ', body)
    body = re.sub(r'\\begin\s*{[^}]*}\s*(?:\[[^\]]*\]\s*)?', ' ', body)
    body = re.sub(r'\\end\s*{[^}]*}', ' ', body)

    # linebreaks and spacing
    body = re.sub(r'(?:\\\\\s*(?:\[[^\]]*\]\s*)?|\\par\b|\\newline\b|\\linebreak\b|\\break\b|\\pagebreak\b|\\smallskip\b|\\medskip\b|\\bigskip\b|\\\s+)', ' ', body)
    body = re.sub(r'\\(?:vspace|hspace)\*?\s*(?:\[[^\]]*\]\s*)?\{(?:[^{}]|\{[^{}]*\})*\}', ' ', body)
    body = re.sub(r'\\\\-\s*|\\-\s*', '', body)

    # special commands
    body = strip_tikzstyle_defs(body)
    body = strip_arg_cmds(body, {r'\acknowledgements': ' ', r'\acknowledgments': ' ', r'\tikzcdset': ' ', r'\tikzset': ' ', r'\pgfplotsset': ' ', r'\lstset': ' ', r'\fvset': ' ', r'\ytableausetup': ' ', r'\hypersetup': ' '})
    body = re.sub(r'\\(?:\\\\|\$|[,%&$#_{};:!~^/])', ' ', body)
    body = re.sub(r'\\(?:pgfsys@(?:invoke|literal)|pgfsys@[\w@]+|special|pdfliteral|pdfextension\s+literal)\s*\{[\s\S]*?\}', ' @@fig@@ ', body)
    body = re.sub(r'\\(?:setlength|addtolength|newtheorem|numberwithin)\s*\{[^}]*\}\s*\{[^}]*\}', ' ', body)
    body = re.sub(r'\\(?:itemsep|topsep|parsep|partopsep)\s*-?\d+(?:\.\d+)?\s*(?:pt|mm|cm|in|em|ex)\b', ' ', body)
    body = re.sub(r'\\(?:setpartpreamble|titleformat)\s*\[[^\]]*\]\s*\[[^\]]*\]\s*', ' ', body)
    body = re.sub(r'\\(?:RedeclareSectionCommand|DeclareSectionCommand|ProvideSectionCommand|setkomafont|addtokomafont)\s*\[[^\]]*\]\s*\{[^}]*\}\s*', ' ', body)
    body = re.sub(r'\\[A-Za-z@]+\s*=\s*(?:-?\d+\b|-?\d+(?:\.\d+)?\s*(?:pt|mm|cm|in|em|ex)\b)', ' ', body)
    body = re.sub(r'\\rotatebox\s*(?:\[[^\]]*\]\s*)?\{[^{}]*\}\s*', ' ', body)
    body = re.sub(r'\\setcounter\{[^}]+\}\{[^}]+\}', ' ', body)
    body = re.sub(r'\\textcolor\s*\{[^{}]*\}\s*', '', body)
    body = re.sub(r'\\[0-9]+', ' ', body)
    body = re.sub(r'\\(?![A-Za-z@])', ' ', body)
    body = re.sub(r'\\[A-Za-z@]+\*?', ' ', body)
    body = re.sub(r'\[\s*\]', ' ', body)
    body = re.sub(r'[{}]', '', body)

    # grunt numbers, citation ids, and other number patterns (except some "good" ones)
    body = re.sub(r'\*?\b[A-Za-z]+:\d{4}[A-Za-z0-9]*\b', ' @@cite@@ ', body)
    body = re.sub(r'(?<![0-9A-Za-z])(?=[0-9A-Za-z\/\.]*[A-Za-z])(?=[0-9A-Za-z\/\.]*\d)[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*(?:/[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)+(?![0-9A-Za-z])', ' @@num@@ ', body)
    body = re.sub(r'(?<![0-9A-Za-z])(?!(?:(?:[A-Za-z]{1,3}(?:0|1|2|3|4|5|6|7|8|9|10|11|26))|(?:(?:0|1|2|3|4|5|6|7|8|9|10|11|26)[A-Za-z])|(?:0|1|2|3|4|5|6|7|8|9|10|11|26))(?![0-9A-Za-z\.]))[0-9A-Za-z]*\d[0-9A-Za-z]*(?:\.[0-9A-Za-z]+)*(?![0-9A-Za-z])', ' @@num@@ ', body)

    # punctuation fixes and placeholder formatting
    body = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE63\uFF0D]", "-", body)
    body = re.sub(r'([\-\/:.,;!?=])\1+', r'\1', body)
    body = re.sub(r'\s*=\s*', ' = ', body)
    body = re.sub(r',\s*,+', ',', body)
    body = re.sub(r'[.,]*\.', '.', body)
    body = re.sub(r'(?:(?<=\S)-(?=\s)|(?<=\s)-(?=\S))', ' ', body)
    for _ in range(3):
        body = re.sub(r'([\(\[])\s+(@@[a-z]+@@)', r'\1\2', body)
        body = re.sub(r'(@@[a-z]+@@)\s+([.,:;!?\)\]])', r'\1\2', body)
        body = re.sub(r'(@@[a-z]+@@)\s*(?:,|:|;|-|/)\s*(?=@@[a-z]+@@)', r'\1 ', body)
        body = re.sub(r'(@@[a-z]+@@)\s*,?\s*\b(?:and|or)\b\s*(?=@@[a-z]+@@)', r'\1 ', body)
        body = re.sub(r'@@(?:ref|cite)@@\s+@@(eqn|math|fig|code|alg|num|url)@@', r'@@\1@@', body)
        body = re.sub(r'@@(eqn|math|fig|code|alg|num|url)@@\s+@@(?:ref|cite)@@', r'@@\1@@', body)
        body = re.sub(r'@@url@@\s+@@(eqn|math|fig|code|alg|num)@@', r'@@\1@@', body)
        body = re.sub(r'@@(eqn|math|fig|code|alg|num)@@\s+@@url@@', r'@@\1@@', body)
        body = re.sub(r'@@num@@\s+@@(eqn|math|fig|code|alg)@@', r'@@\1@@', body)
        body = re.sub(r'@@(eqn|math|fig|code|alg)@@\s+@@num@@', r'@@\1@@', body)
        body = re.sub(r'@@alg@@\s+@@(eqn|math|fig|code)@@', r'@@\1@@', body)
        body = re.sub(r'@@(eqn|math|fig|code)@@\s+@@alg@@', r'@@\1@@', body)
        body = re.sub(r'@@code@@\s+@@(eqn|math|fig)@@', r'@@\1@@', body)
        body = re.sub(r'@@(eqn|math|fig)@@\s+@@code@@', r'@@\1@@', body)
        body = re.sub(r'@@fig@@\s+@@(eqn|math)@@', r'@@\1@@', body)
        body = re.sub(r'@@(eqn|math)@@\s+@@fig@@', r'@@\1@@', body)
        body = re.sub(r'@@math@@\s+@@eqn@@', r'@@eqn@@', body)
        body = re.sub(r'@@eqn@@\s+@@math@@', r'@@eqn@@', body)
        body = re.sub(r'(@@[a-z]+@@)(?:\s+@@[a-z]+@@)+', r'\1', body)
        body = re.sub(r'[\(\[]\s*(@@[a-z]+@@)\s*[\)\]]', r'\1', body)
        body = re.sub(r'([\(\[])\s+(@@[a-z]+@@)', r'\1\2', body)
        body = re.sub(r'(@@[a-z]+@@)\s+([.,:;!?\)\]])', r'\1\2', body)

    # pad placeholders glued to words or brackets
    body = re.sub(r'(\w)(@@[a-z]+@@)', r'\1 \2', body)
    body = re.sub(r'(@@[a-z]+@@)(?=\w|[\(\[])', r'\1 ', body)

    # final punctuation spacing
    body = re.sub(r'\s+([.,:;)\]])', r'\1', body)
    body = re.sub(r'([\[(])\s+', r'\1', body)

    # collapse whitespace
    body = re.sub(r'\s+', ' ', body)

    # case
    body = body.lower()
    body = re.sub(r'@@([a-z]+)@@', lambda m: m.group(1).upper(), body)

    return latinize(body.strip())

def extract_sections(latex):
    matches = []
    for cmd in ['chapter', 'section', 'paragraph']:
        sec_re = re.compile(r'\\(?:' + cmd + r')\s*\*?\s*(?:\[[^\]]*\]\s*)?', flags = re.IGNORECASE)
        matches = []
        pos = 0
        while True:
            m = sec_re.search(latex, pos)
            if not m:
                break
            title, j = read_balanced(latex, m.end(), "{", "}")
            if title is None:
                pos = m.end()
                continue
            matches.append((m.start(), j, title))
            pos = j
        if matches:
            break
    if not matches:
        return [{'title': '', 'text': re.sub(r'\s+', ' ', (latex or '').strip())}]
    skip_re = re.compile(r'\b(?:ac?know[a-z]*ment(?:s)?|dedication(?:s)?|abstract(?:s)?)\b', flags = re.IGNORECASE)
    stop_re = re.compile(r'\b(?:reference(?:s)?|bibliograph(?:y|ies)|bibliorgraph(?:y|ies)|appendix(?:es)?|appendices)\b', flags = re.IGNORECASE)
    sections = []
    for i, (start_i, title_end_i, title) in enumerate(matches):
        if stop_re.search(title or ''):
            break
        if skip_re.search(title or ''):
            continue
        start = title_end_i
        end = matches[i + 1][0] if i + 1 < len(matches) else len(latex)
        content = latex[start:end].strip()
        content = re.sub(r'\s+', ' ', content)
        sections.append({'title': title.strip(), 'text': content})
    if not sections:
        return [{'title': '', 'text': re.sub(r'\s+', ' ', (latex or '').strip())}]
    return sections

def unprocessed_math_symbol_count(text, symbols = "$_^&\\"):
    if not text:
        return 0
    text = re.sub(r'\\section\*?\s*(?:\[[^\]]*\]\s*)?{[^}]*}', ' ', text)
    bad = sum(text.count(ch) for ch in symbols)
    return bad

def expand_inputs(latex, base_dir, max_depth = 8):
    input_re = re.compile(r'\\(?:input|include|subfile)\s*{([^}]+)}')
    visited = set()

    def normpath(p):
        p = (p or "").strip()
        p = p.replace("\\", "/")
        p = p.split("%", 1)[0].strip()
        return p

    def read_tex(path):
        try:
            with open(path, "r", encoding = "utf-8", errors = "ignore") as f:
                return f.read()
        except Exception:
            return ""

    def resolve(name, cur_dir):
        name = normpath(name)
        if not name:
            return None
        cand = os.path.join(cur_dir, name)
        if os.path.exists(cand) and os.path.isfile(cand):
            return cand
        if not os.path.splitext(cand)[1]:
            cand2 = cand + ".tex"
            if os.path.exists(cand2) and os.path.isfile(cand2):
                return cand2
        return None

    def rec(text, cur_dir, depth):
        if depth <= 0:
            return text

        def repl(m):
            rel = m.group(1)
            path = resolve(rel, cur_dir)
            if path is None:
                return " "
            path_abs = os.path.abspath(path)
            if path_abs in visited:
                return " "
            visited.add(path_abs)
            content = read_tex(path_abs)
            if not content:
                return " "
            next_dir = os.path.dirname(path_abs)
            return "\n" + rec(content, next_dir, depth - 1) + "\n"

        while True:
            new = input_re.sub(repl, text)
            if new == text:
                break
            text = new
        return text

    return rec(latex, base_dir, max_depth)

def process_sources(data_dir = "../data", skip_threshold = 5, overwrite = False, max_papers = None):
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok = True)
    meta = load_metadata(os.path.join(data_dir, "metadata.jsonl"))
    skipped_path = os.path.join(processed_dir, "skipped.jsonl")
    total = 0
    skipped = 0

    ids = [bid for bid in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, bid))]
    n_ids = len(ids)
    target_total = min(max_papers, n_ids) if max_papers is not None else n_ids

    for i, base_id in enumerate(ids, 1):
        paper_dir = os.path.join(raw_dir, base_id)
        if not os.path.isdir(paper_dir):
            continue
        if max_papers is not None and total >= max_papers:
            break
        total += 1
        out_path = os.path.join(processed_dir, f"{base_id}.json")
        if not overwrite and os.path.exists(out_path):
            pct = 100 * total / target_total if target_total else 0
            print(f"\rprogress: {pct:6.2f}%", end = "", flush = True)
            continue
        if base_id not in meta:
            pct = 100 * total / target_total if target_total else 0
            print(f"\rprogress: {pct:6.2f}%", end = "", flush = True)
            continue
        tar_path = os.path.join(paper_dir, "source.tar.gz")
        if not os.path.exists(tar_path):
            pct = 100 * total / target_total if target_total else 0
            print(f"\rprogress: {pct:6.2f}%", end = "", flush = True)
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(tmpdir)
            except Exception:
                pct = 100 * total / target_total if target_total else 0
                print(f"\rprogress: {pct:6.2f}%", end = "", flush = True)
                continue
            main_tex = None
            for root, _, files in os.walk(tmpdir):
                for f_name in files:
                    if not f_name.endswith(".tex"):
                        continue
                    full = os.path.join(root, f_name)
                    try:
                        with open(full, "r", encoding = "utf-8", errors = "ignore") as f:
                            txt = f.read()
                    except Exception:
                        continue
                    if re.search(r'\\begin\s*\{\s*document\s*\}', txt, flags = re.IGNORECASE):
                        main_tex = full
                        break
                if main_tex:
                    break
            if not main_tex:
                pct = 100 * total / target_total if target_total else 0
                print(f"\rprogress: {pct:6.2f}%", end = "", flush = True)
                continue
            try:
                with open(main_tex, "r", encoding = "utf-8", errors = "ignore") as f:
                    latex = f.read()
            except Exception:
                pct = 100 * total / target_total if target_total else 0
                print(f"\rprogress: {pct:6.2f}%", end = "", flush = True)
                continue
            latex = expand_inputs(latex, os.path.dirname(main_tex))
            macros = extract_macros(latex, package = False)
            for sty_path in find_custom_sty_packages(latex, tmpdir):
                try:
                    with open(sty_path, "r", encoding = "utf-8", errors = "ignore") as f:
                        sty_txt = f.read()
                except Exception:
                    continue
                macros = merge_macros(macros, extract_macros(sty_txt, package = True))
            match = re.search(r'\\begin\s*{document}([\s\S]*?)\\end\s*{document}', latex, re.IGNORECASE)
            body = match.group(1) if match else latex
            body = replace_macros(body, macros)
            m_end_titlepage = re.search(r'\\end\s*\{titlepage\}', body, re.IGNORECASE)
            if m_end_titlepage:
                body = body[m_end_titlepage.end():]
            else:
                m_maketitle = re.search(r'\\maketitle\b', body, re.IGNORECASE)
                if m_maketitle:
                    body = body[m_maketitle.end():]
            body = re.sub(r'(?<!\\)%.*', '', body)
            body = re.sub(r'[A-Za-z0-9]*\^\^[A-Za-z0-9]*', ' ', body)
            body = re.sub(r'(?is)\\(?:bibliographystyle|bibliography|bibstyle|bibdata|citation)\b\s*(?:\[[^\]]*\]\s*)?\{(?:[^{}]|\{[^{}]*\})*\}', ' ', body)
            body = re.sub(r'(?is)\\(?:bibcite|newlabel|@input|@writefile)\b\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*\{(?:[^{}]|\{[^{}]*\})*\}', ' ', body)
            body = re.sub(r'(?is)\\begin\s*\{(?:thebibliography|references)\}[\s\S]*?\\end\s*\{(?:thebibliography|references)\}', '', body)
            body = re.sub(r'(?is)\\begin\s*\{acknowledgments\}\s*\*?\s*[\s\S]*?\\end\s*\{acknowledgments\}', '', body)
            body = re.sub(r'(?is)\\begin\s*\{appendix\}[\s\S]*\Z', '', body)
            body = re.sub(r'(?is)\\appendix\b[\s\S]*\Z', '', body)
            sections = extract_sections(body)
            if not sections or (len(sections) == 1 and (sections[0].get("title") or "") == ""):
                sections = [{"title": "", "text": body}]
            cleaned_sections = []
            for s in sections:
                t = s.get("title") or ""
                x = s.get("text") or ""
                x = clean_body(x)
                cleaned_sections.append({"title": t, "text": x})
            full_text = " ".join(s["text"] for s in cleaned_sections)
            count = unprocessed_math_symbol_count(full_text)
            if count > skip_threshold:
                with open(skipped_path, "a", encoding = "utf-8") as sk_f:
                    sk_f.write(json.dumps({"arxiv_id": base_id, "math_symbol_count": count}, ensure_ascii = False) + "\n")
                skipped += 1
                pct = 100 * total / target_total if target_total else 0
                print(f"\rprogress: {pct:6.2f}%", end = "", flush = True)
                continue
            record = {
                "arxiv_id": base_id,
                "title": meta[base_id].get("title"),
                "authors": meta[base_id].get("authors"),
                "abstract": meta[base_id].get("abstract"),
                "categories": meta[base_id].get("categories"),
                "dates": meta[base_id].get("dates"),
                "sections": cleaned_sections,
            }
            with open(out_path, "w", encoding = "utf-8") as out_f:
                json.dump(record, out_f, ensure_ascii = False, indent = 2)

        pct = 100 * total / target_total if target_total else 0
        print(f"\rprogress: {pct:6.2f}%", end = "", flush = True)

    successful = total - skipped
    if total > 0:
        p_success = round(100 * successful / total, 2)
        p_skipped = round(100 * skipped / total, 2)
    else:
        p_success = 0
        p_skipped = 0

    print("\r" + " " * 30 + "\r", end = "")
    print("\ntotal papers processed:", total)
    print(f"successfully processed: {successful} ({p_success} %)")
    print(f"skipped: {skipped} ({p_skipped} %)")

def build_corpus(
    data_dir = "../data",
    chunk_size = 1500,
    overlap = 150,
    placeholder_threshold = 0.15,
    min_tokens = 32,
    overwrite = False  
):
    processed_dir = os.path.join(data_dir, "processed")
    corpus_dir = os.path.join(data_dir, "corpus")
    os.makedirs(corpus_dir, exist_ok = True)
    chunks_path = os.path.join(corpus_dir, "chunks.jsonl")
    if not overwrite and os.path.exists(chunks_path):
        print("Chunks file already exists. Use overwrite = True to rebuild.")
        return
    skipped = set()
    skipped_path = os.path.join(processed_dir, "skipped.jsonl")
    if os.path.exists(skipped_path):
        with open(skipped_path, "r", encoding = "utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                arxiv_id = rec.get("arxiv_id")
                if arxiv_id:
                    skipped.add(arxiv_id)
    overlap_local = overlap if overlap > 0 else 0
    if chunk_size <= 0:
        overlap_local = 0
    elif overlap_local >= chunk_size:
        overlap_local = chunk_size - 1
    min_tail = max(300, chunk_size // 6)
    token_re = re.compile(r'\S+')
    placeholders = {"MATH", "NUM", "EQN", "CITE", "REF", "FIG", "URL", "CODE", "ALG", "ENV"}
    strip_chars = ".,;:!?()[]{}<>\"'`"

    def split_text(text):
        text = text.strip()
        if chunk_size <= 0 or len(text) <= chunk_size:
            return [(text, 0)]
        chunks = []
        n = len(text)
        start = 0
        while start < n:
            nominal_end = start + chunk_size
            if nominal_end >= n:
                chunk_text = text[start:n].strip()
                if chunk_text:
                    chunks.append(chunk_text)
                break
            cut = text.rfind(' ', start, nominal_end)
            if cut <= start:
                cut2 = text.find(' ', nominal_end)
                cut = n if cut2 == -1 else cut2
            chunk_text = text[start:cut].strip()
            if chunk_text:
                chunks.append(chunk_text)
            if cut >= n:
                break
            next_start = cut - overlap_local
            if next_start < 0:
                next_start = 0
            if next_start > 0 and not text[next_start].isspace() and not text[next_start - 1].isspace():
                sp = text.find(' ', next_start, cut)
                next_start = cut if sp == -1 else sp
            while next_start < n and text[next_start].isspace():
                next_start += 1
            if next_start <= start:
                next_start = cut
                while next_start < n and text[next_start].isspace():
                    next_start += 1
            start = next_start
        if len(chunks) >= 2 and len(chunks[-1]) < min_tail:
            tail = chunks[-1]
            if overlap_local > 0:
                if len(tail) <= overlap_local:
                    tail = ""
                else:
                    tail = tail[overlap_local:].lstrip()
            if tail:
                chunks[-2] = (chunks[-2] + ' ' + tail).strip()
            chunks.pop()
        return [(chunk, i) for i, chunk in enumerate(chunks)]
    
    files = [fn for fn in os.listdir(processed_dir) if fn.endswith(".json")]
    files.sort()
    added_chunks = 0
    total_chunks = 0

    with open(chunks_path, "w", encoding = "utf-8") as ch_out:
        write_line = ch_out.write
        for fname in files:
            with open(os.path.join(processed_dir, fname), "r", encoding = "utf-8") as f:
                record = json.load(f)
            doc_id = record.get("arxiv_id")
            if not doc_id or doc_id in skipped:
                continue
            title = record.get("title")
            authors = record.get("authors")
            categories = record.get("categories")
            dates = record.get("dates")
            sections = record.get("sections", [])
            chunk_count = 0
            for sec_idx, section in enumerate(sections):
                sec_text = section.get("text", "")
                if not sec_text:
                    continue
                sec_title = section.get("title", "")
                for chunk_text, local_idx in split_text(sec_text):
                    total_chunks += 1
                    tok_len = len(_WORD_RE.findall(chunk_text))
                    if min_tokens is not None and tok_len < min_tokens:
                        continue
                    if placeholder_threshold is not None:
                        n_tokens = 0
                        n_placeholders = 0
                        for m in token_re.finditer(chunk_text):
                            tok = m.group(0).strip(strip_chars)
                            if not tok:
                                continue
                            tok = tok.upper()
                            n_tokens += 1
                            if tok in placeholders:
                                n_placeholders += 1
                        if n_tokens > 0 and (n_placeholders / n_tokens) > placeholder_threshold:
                            continue
                    chunk_record = {
                        "chunk_id": f"{doc_id}_s{sec_idx}_c{local_idx}",
                        "doc_id": doc_id,
                        "title": title,
                        "section": sec_title,
                        "text": chunk_text,
                        "position": chunk_count,
                        "authors": authors,
                        "categories": categories,
                        "dates": dates,
                    }
                    write_line(json.dumps(chunk_record, ensure_ascii = False) + "\n")
                    chunk_count += 1
                    added_chunks += 1

    discarded_chunks = total_chunks - added_chunks
    if total_chunks > 0:
        p_added = round(100 * added_chunks / total_chunks, 2)
        p_discarded = round(100 * discarded_chunks / total_chunks, 2)
    else:
        p_added = 0
        p_discarded = 0

    print("\ntotal number of chunks:", total_chunks)
    print(f"chunks added to corpus: {added_chunks} ({p_added} %)")
    print(f"chunks discarded: {discarded_chunks} ({p_discarded} %)")

def estimate_sequence_length(corpus_dir = "../data/corpus", percentile = 0.95, multiple_of = 32, verbose = True):
    def percentile_value(sorted_vals, p):
        if p <= 0:
            return sorted_vals[0]
        if p >= 1:
            return sorted_vals[-1]
        k = int(p * (len(sorted_vals) - 1))
        return sorted_vals[k]
    
    chunks_path = os.path.join(corpus_dir, "chunks.jsonl")
    token_re = re.compile(r'\S+')
    token_lens = []
    with open(chunks_path, "r", encoding = "utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text", "")
            if not text:
                continue
            n = 0
            for _ in token_re.finditer(text):
                n += 1
            token_lens.append(n)
    token_lens.sort()
    target = percentile_value(token_lens, percentile)
    if multiple_of is None:
        suggested_seq_len = int(target)
    else:
        suggested_seq_len = int(((target + (multiple_of - 1)) // multiple_of) * multiple_of)
    if verbose:
        print("suggested sequence length:", suggested_seq_len)
    return suggested_seq_len