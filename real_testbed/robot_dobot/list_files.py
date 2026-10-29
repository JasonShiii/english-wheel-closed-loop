"""print {relative path: [size, mtime]} of every file under a folder, plus <folder>.log next to it (for pull_trial --follow)"""
import json, os, sys
root = sys.argv[1]; out = {}
for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f)
        try:
            st = os.stat(p); out[os.path.relpath(p, root).replace("\\", "/")] = [st.st_size, int(st.st_mtime)]
        except OSError:
            pass
log = root.rstrip("\\/") + ".log"
if os.path.exists(log):
    st = os.stat(log); out["../" + os.path.basename(log)] = [st.st_size, int(st.st_mtime)]
print(json.dumps(out))
