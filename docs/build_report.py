"""docs/build_report.py

Substitute <!--CODE:...--> placeholders in report.html with real source, pulled
from the repository and HTML-escaped.

WHY THE REPORT DOES NOT CONTAIN PASTED CODE
    A report with pasted code is wrong the moment the code changes, and nothing
    tells you. Pulling every block from the repository at build time means the
    document cannot drift from the software it documents.

WHY THE PLACEHOLDERS ARE NO LONGER LINE NUMBERS  (R4)
    They were, and it was the wrong mechanism -- for exactly the same reason.
    A line-range placeholder silently starts quoting the wrong function as soon
    as anything above it grows by a line, and the build still succeeds. The
    failure is a report that confidently shows the wrong code.

    Placeholders now name a SYMBOL, and the build fails loudly if the symbol
    cannot be found:

        <!--CODE:path:py:function_or_class-->    Python def/class, by name
        <!--CODE:path:cpp:function-->            C/C++ function, brace-matched
        <!--CODE:path:sect:BANNER TEXT-->        a ==== banner ==== section
        <!--CODE:path:region:TAG-->              between [[TAG]] and [[/TAG]]
        <!--CODE:path:lines:12:40-->             explicit lines, last resort

    The py: and cpp: forms include the doc comment immediately above the
    definition, because in this codebase the reasoning lives there and quoting
    a function without it quotes half of it.

    python docs/build_report.py
"""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"<!--CODE:([^:]+):([a-z]+):([^>]+)-->")


def _lead_comment(lines, i):
    """Walk back over the contiguous comment block directly above line i."""
    j = i
    while j > 0:
        prev = lines[j - 1].strip()
        if prev.startswith("//") or prev.startswith("#"):
            j -= 1
        elif prev.startswith("@"):        # a Python decorator
            j -= 1
        else:
            break
    # Do not swallow a banner rule that belongs to the section above.
    while j < i and re.match(r"^\s*(//|#)\s*[-=]{5,}\s*$", lines[j]):
        j += 1
    return j


def _scrub(line):
    """Blank out strings, chars and line comments so brace counting is honest."""
    line = re.sub(r"//.*$", "", line)
    line = re.sub(r'"(\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(\\.|[^'\\])*'", "''", line)
    return line


def extract_py(lines, name):
    pat = re.compile(rf"^(\s*)(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b")
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if not m:
            continue
        indent = len(m.group(1))
        j = i + 1
        end = j
        while j < len(lines):
            s = lines[j]
            if s.strip() and (len(s) - len(s.lstrip())) <= indent:
                break
            if s.strip():
                end = j
            j += 1
        return _lead_comment(lines, i), end + 1
    return None


def extract_cpp(lines, name):
    # A definition, not a declaration: the signature line mentions the name with
    # an open paren, and an opening brace arrives within the next three lines.
    pat = re.compile(rf"(^|[\s*&:~])({re.escape(name)})\s*\(")
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("//") or not pat.search(_scrub(ln)):
            continue
        head = "".join(_scrub(x) for x in lines[i:i + 4])
        if "{" not in head or ";" in head.split("{")[0]:
            continue                       # a declaration, or a call site
        depth = 0
        started = False
        for j in range(i, len(lines)):
            s = _scrub(lines[j])
            depth += s.count("{") - s.count("}")
            if "{" in s:
                started = True
            if started and depth <= 0:
                return _lead_comment(lines, i), j + 1
        break
    return None


def extract_sect(lines, title):
    banner = re.compile(r"^\s*(//|#|;)\s*[=]{3,}")
    start = None
    for i, ln in enumerate(lines):
        if banner.match(ln) and title.lower() in ln.lower():
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        if banner.match(lines[j]):
            return start, j
    return start, len(lines)


def extract_region(lines, tag):
    a = b = None
    for i, ln in enumerate(lines):
        if f"[[{tag}]]" in ln:
            a = i + 1
        elif f"[[/{tag}]]" in ln:
            b = i
    return (a, b) if a is not None and b is not None else None


def sub(m):
    path, mode, arg = m.group(1), m.group(2), m.group(3)
    src = (ROOT / path)
    if not src.exists():
        sys.exit(f"FATAL: {path} does not exist (placeholder {m.group(0)})")
    lines = src.read_text().splitlines()

    if mode == "lines":
        a, b = (int(v) for v in arg.split(":"))
        span = (a - 1, b)
    elif mode == "py":
        span = extract_py(lines, arg)
    elif mode == "cpp":
        span = extract_cpp(lines, arg)
    elif mode == "sect":
        span = extract_sect(lines, arg)
    elif mode == "region":
        span = extract_region(lines, arg)
    else:
        sys.exit(f"FATAL: unknown extraction mode '{mode}' in {m.group(0)}")

    if span is None:
        sys.exit(f"FATAL: could not find {mode}:{arg} in {path}. "
                 f"The symbol was renamed or removed; fix the placeholder rather "
                 f"than the report text around it.")

    chunk = lines[span[0]:span[1]]
    while chunk and not chunk[0].strip():
        chunk.pop(0)
    while chunk and not chunk[-1].strip():
        chunk.pop()
    if not chunk:
        sys.exit(f"FATAL: {mode}:{arg} in {path} extracted to nothing")

    body = html.escape("\n".join(chunk))
    print(f"  {path:<44} {mode}:{arg:<26} {len(chunk):>4} lines")
    cls = "long" if len(chunk) > 34 else ""
    return (f'<div class="filename">{html.escape(path)}</div>\n'
            f'<pre class="{cls}">{body}</pre>')


def main():
    report = ROOT / "docs" / "report.html"
    print("extracting code blocks:")
    out = PAT.sub(sub, report.read_text())
    left = re.findall(r"<!--CODE:[^>]*-->", out)
    if left:
        sys.exit(f"unsubstituted placeholders: {left}")
    dest = ROOT / "docs" / "report_built.html"
    dest.write_text(out)
    print(f"\nwrote {dest.relative_to(ROOT)} ({len(out):,} bytes)")


if __name__ == "__main__":
    main()
