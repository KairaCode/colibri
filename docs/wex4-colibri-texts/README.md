# Texts of the WEX series on JustVugg/colibri

The tracking issue and the 20 pull requests of the series were closed upstream on 2026-11-17; their code is the branch `kaira/wex4-colibri` (see `docs/wex4-colibri-series.md`). These files keep their texts:

- [issue-2860.md](issue-2860.md): the tracking issue, including "Tests wanted (help welcome)"
- [pr-2861.md](pr-2861.md): #2861: wex: the WEX record format headers (wex_format.h) and their test
- [pr-2862.md](pr-2862.md): #2862: wex: the WEX4 container reader (wex4.h) and tools/wex4_verify
- [pr-2863.md](pr-2863.md): #2863: deepseek-v4: routed experts from a WEX4 container (COLI_WEX_PATH, COLI_WEX_ROOT)
- [pr-2864.md](pr-2864.md): #2864: qwen38: MXFP4 experts from a WEX4 container (COLI_WEX_PATH)
- [pr-2865.md](pr-2865.md): #2865: deepseek_v41: routed experts from a WEX4 container (COLI_WEX_PATH, COLI_WEX_ROOT)
- [pr-2876.md](pr-2876.md): #2876: wex: WNR3 records on the CPU (wex_decode.c), the WNM2 model registry
- [pr-2877.md](pr-2877.md): #2877: deepseek-v4: WNR3 records from a WEX4 container, decoded on the CPU
- [pr-2878.md](pr-2878.md): #2878: qwen38: WNR3 records from a WEX4 container, decoded on the CPU
- [pr-2879.md](pr-2879.md): #2879: deepseek_v41: WNR3 records from a WEX4 container, decoded on the CPU
- [pr-2880.md](pr-2880.md): #2880: tools: the WNR3 encoder (wex_nr3_tools.c, build_nr3.py)
- [pr-2883.md](pr-2883.md): #2883: wex: the CUDA helper for WNR3 records (tools/wex_cuda_bridge.cu, make wex-cuda-helper) and its GPU test
- [pr-2884.md](pr-2884.md): #2884: wex: COLI_WEX_CUDA_LIB loads the CUDA helper; WNR3 records decode on the GPU through coli_wex_decode_record
- [pr-2885.md](pr-2885.md): #2885: qwen38, deepseek_v41: a WEX4 container read from COLI_MODEL_MIRROR copies, split by COLI_DISK_WEIGHTS
- [pr-2886.md](pr-2886.md): #2886: deepseek_v41: a batch's first-read SHA-256 on the OpenMP team (COLI_WEX_VERIFY)
- [pr-2888.md](pr-2888.md): #2888: wex: a fast CPU decoder for WNR3 records (wex_fast_decode.h) and its test against the reference
- [pr-2889.md](pr-2889.md): #2889: wex: WNR3 records decode on the CPU with the fast decoder (COLI_WEX_FAST_DECODE), AVX2 or NEON chosen at run time
- [pr-2894.md](pr-2894.md): #2894: wex: tools/wex_bench.py, one command for the WEX measurements helpers are asked for
- [pr-2895.md](pr-2895.md): #2895: glm53: AMD MXFP4 checkpoint as published (amd/GLM-5.3-Flash-Quark-MXFP4), experts as fmt 7
- [pr-2896.md](pr-2896.md): #2896: glm53: MXFP4 routed experts from a WEX4 container (COLI_WEX_PATH, COLI_WEX_ROOT), RAW and WNR3
- [pr-2897.md](pr-2897.md): #2897: wex: WNR3 records decoded straight into the expert's buffers (coli_wex_decode_record_to), the fast decoder's stream count measured (COLI_WEX_FAST_LANES); make check green on the WEX stack

Comments by other people are linked, not copied.
