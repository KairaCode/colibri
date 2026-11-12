set -u
cd c
rc=0
python3 -m pip install -q numpy >/dev/null 2>&1 || true
for m in tests.test_resource_plan tests.test_setup_hw tests.test_openai_server; do
  echo "::group::$m"; python3 -m unittest -q "$m" || { echo "FAILED: $m"; rc=1; }; echo "::endgroup::"
done
echo "== real macOS memory readings (for the PR text)"
vm_stat | head -16; sysctl hw.memsize vm.page_free_count vm.compressor_page_count 2>/dev/null || true
python3 - <<'PY' || rc=1
import sys; sys.path.insert(0, '.')
import resource_plan as rp
for name in dir(rp):
    if 'avail' in name.lower() and callable(getattr(rp, name)):
        try: print(name, getattr(rp, name)())
        except TypeError: pass
PY
exit $rc
