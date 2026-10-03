"""Check shell scripts for constructs that break on macOS bash 3.2."""
import re, sys
from pathlib import Path
RULES = [
    (re.compile(rb'\$[A-Za-z_][A-Za-z0-9_]*(?=[\x80-\xff])'), "unbraced $VAR followed by a UTF-8 byte (bash 3.2 reads it as part of the name)"),
    (re.compile(rb'declare\s+-A|local\s+-A'), "associative arrays (bash 4+)"),
    (re.compile(rb'\b(mapfile|readarray)\b'), "mapfile/readarray (bash 4+)"),
    (re.compile(rb'\$\{[^}]*(,,|\^\^)\}'), "case modification ${v,,} (bash 4+)"),
    (re.compile(rb'\|&|&>>'), "|& or &>> (bash 4+)"),
    (re.compile(rb'"\$\{[A-Za-z_]+\[@\]\}"'), "\"${arr[@]}\" errors under set -u when empty (bash < 4.4)"),
    (re.compile(rb'(?<!\{1\+)"\$@"'), "\"$@\" under set -u: use ${1+\"$@\"} (bash 3.2)"),
]
bad = 0
for f in sys.argv[1:]:
    for n, line in enumerate(Path(f).read_bytes().split(b"\n"), 1):
        if line.lstrip().startswith(b"#"):
            continue
        for rx, msg in RULES:
            if rx.search(line):
                # "$@" inside a function body that is called with args is fine (native "$@")
                if b'"$@"' in line and (b'arch -arm64 "$@"' in line or b'else "$@"' in line or b"say()" in line or b"die()" in line):
                    continue
                print(f"{f}:{n}: {msg}\n    {line.decode('utf-8','replace').strip()}")
                bad += 1
print("lint:", "OK" if not bad else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
