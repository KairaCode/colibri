set -u
cd c
rc=0
python3 -m pip install -q numpy >/dev/null 2>&1 || true
for m in tests.test_resource_plan tests.test_setup_hw tests.test_openai_server; do
  echo "::group::$m"; python3 -m unittest -q "$m" || { echo "FAILED: $m"; rc=1; }; echo "::endgroup::"
done
echo "== real macOS memory readings (for the PR text)"
sw_vers; vm_stat | head -16; sysctl hw.memsize vm.page_free_count vm.compressor_page_count 2>/dev/null || true
python3 - <<'PY' || rc=1
import sys; sys.path.insert(0, '.')
import resource_plan as rp
for name in dir(rp):
    if 'avail' in name.lower() and callable(getattr(rp, name)):
        try: print(name, getattr(rp, name)())
        except TypeError: pass
PY
echo "== check 1, the PR text's one-liner, with the system awk ($(command -v awk))"
vm_stat | awk '/page size of/ {p=$8} /^Pages free/ {f=$3} /^File-backed pages/ {b=$3}
  /^Pages purgeable/ {g=$3} /^Pages inactive/ {i=$3} /^Pages speculative/ {s=$3}
  END {printf "free+file-backed+purgeable %.2f GiB (old free+inactive+speculative+purgeable %.2f GiB)\n", (f+b+g)*p/2^30, (f+i+s+g)*p/2^30;
       printf "speculative %.2f GiB, the most the first figure can leave out\n", s*p/2^30}' || { echo "FAILED: check 1 awk"; rc=1; }
echo "== control: gawk's 3-argument match() under the system awk (expected to fail on BSD awk)"
echo x | awk '{ match($0, /x/, m) }' && echo "accepted" || echo "rejected rc=$?"
echo "== check 2, Python and C"
python3 -c 'import sys; sys.path.insert(0, "."); import resource_plan; print(resource_plan.memory_available() / 1e9, "GB (Python)")' || rc=1
printf '#include <stdio.h>\n#include "compat.h"\nint main(void){ double t, a; compat_meminfo_gb(&t, &a);\n  printf("%%.2f GB (C)\\n", a); return 0; }\n' > /tmp/avail.c
cc -I . -o /tmp/avail /tmp/avail.c && /tmp/avail || { echo "FAILED: check 2 C"; rc=1; }
exit $rc
