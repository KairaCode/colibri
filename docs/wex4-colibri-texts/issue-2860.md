# Tracking issue JustVugg/colibri#2860, as it read on 2026-11-17

Copied from https://github.com/JustVugg/colibri/issues/2860 so that the fork keeps it. Its "Tests wanted (help welcome)" section is the list of tests still missing.

Postponed --> 1/2 weeks
This issue tracks a series of small pull requests that bring the WEX expert container into colibri. Each PR builds and passes `make test-c` on its own; the later ones are stacked on the earlier ones and say so in their first lines.

## What WEX4 and WNR3 are

- **WEX4** is a container for a MoE model's routed experts: one file, one record per (layer, expert), each record on a 4096-byte page and already in the layout the expert kernels use. A miss is one aligned read of one record instead of six tensor reads across the safetensors shards. The first page holds the SHA-256 of the index, the index holds the SHA-256 of every record, and the hash of the first page (the *root*) names the whole file, so an engine can refuse a file that is not the one it was told to use and check each record the first time it reads it.
- **WNR3** is a lossless codec for an MXFP4 expert record (rANS with a small trained model, a 23-bit check on every stream). A WEX4 record is either RAW or WNR3.

Specifications, conformance vectors and the measurements: [Wex-WRN2](https://github.com/KairaCode/Wex-WRN2) ([WEX4-SPEC](https://github.com/KairaCode/Wex-WRN2/blob/main/docs/WEX4-SPEC.md), [WNR3-SPEC](https://github.com/KairaCode/Wex-WRN2/blob/main/docs/WNR3-SPEC.md), [BENCHMARKS](https://github.com/KairaCode/Wex-WRN2/blob/main/docs/BENCHMARKS.md)).

Measured in a downstream fork of colibri (not on `dev`), DeepSeek V4.1 Flash REAP-192 MXFP4, Windows 11, one desktop:

- RAW container against the safetensors, CPU-only engine, 16 GB RAM, same NVMe drive: decode 1.11-1.14 against 0.98-1.01 tok/s (about +13%), TTFT for 30 tokens 11-12 s against 13-14 s.
- WNR3 makes the 7,680 expert records 0.854 of their RAW size (123.3 GB against 144.4 GB), losslessly. Decoded on the GPU with the container on two drives, one 1,886-token turn of the application gave 2.92-2.96 tok/s against 2.86 for the earlier WNR2 codec, same text; on one drive alone WNR3 was slower (2.15 tok/s).
- A byte copy on a second drive, reads split by a hash of (layer, expert): +49% decode (2.02 to 2.99 tok/s, one pair, RTX 5090, 7 prompts x 64 tokens), bit-identical logits.

None of these numbers has been measured on colibri `dev` yet; each PR says what it measured itself.

## The series

1. **WEX4 container and reader in the engines**: the format headers, the WEX4 reader, `tools/wex4_verify`, and hooks that let DeepSeek V4, DeepSeek V4.1 and Qwen3.8 read RAW expert records from a WEX4 file, with the root pin. CPU and any backend; no GPU code.
2. **WNR3 CPU codec**: the WNR3 model registry and record decode (`wex_decode.c`), WNR3 records accepted by the hooks of series 1, the encoder tool. Opened: #2876-#2880.
3. **GPU decode**: the optional CUDA helper that decodes WNR3 records on the GPU, to host or straight into VRAM.
4. **I/O switches**: mirrors of the container on other drives, split reads, the first-read SHA-256 off the serial path, batched prewarm.
5. **Fast CPU decode**: a SIMD decoder for WNR3 records with the reference's bytes and verdicts, used by `coli_wex_decode_record` on the CPU.

## Checklist

Step 0, merged: Qwen3.8 on AMD Quark's MXFP4 experts (the record layout the Qwen3.8 container uses).

- [x] #2822 tools: convert_qwen38_experts_mxfp4.py writes AMD Quark MXFP4 experts as an experts-mxfp4/ sidecar
- [x] #2823 cuda: MXFP4 (fmt 7) expert groups, one warp per row
- [x] #2824 qwen38: an expert sidecar read from COLI_MODEL_MIRROR copies, split by COLI_DISK_WEIGHTS
- [x] #2825 qwen38: routed experts from an MXFP4 sidecar (Q38_EXPERT_LOWBIT=mxfp4)

Series 1:

- [ ] #2861 wex: the WEX record format headers and their test
- [ ] #2862 wex: the WEX4 container reader and tools/wex4_verify
- [ ] #2863 deepseek-v4: routed experts from a WEX4 container
- [ ] #2864 qwen38: MXFP4 experts from a WEX4 container
- [ ] #2865 deepseek_v41: routed experts from a WEX4 container

Series 2 (WNR3 records on the CPU), stacked on series 1:

- [ ] #2876 wex: WNR3 records on the CPU (wex_decode.c), the WNM2 model registry
- [ ] #2877 deepseek-v4: WNR3 records from a WEX4 container, decoded on the CPU
- [ ] #2878 qwen38: WNR3 records from a WEX4 container, decoded on the CPU
- [ ] #2879 deepseek_v41: WNR3 records from a WEX4 container, decoded on the CPU
- [ ] #2880 tools: the WNR3 encoder (wex_nr3_tools.c, build_nr3.py)

With series 2, the published DeepSeek V4.1 Flash REAP-192 WNR3 container runs end to end on `deepseek_v41` (trunk from the shards): the same 8 tokens as the shards on a 4-core i7-7700K, about 3x slower there, because the reference CPU decoder (24.7 MB/s a core) is the bottleneck (#2879 has the numbers and conditions). No GPU code and no helper loading in series 2.

Series 3 (GPU decode), stacked on series 2:

- [ ] #2883 wex: the CUDA helper for WNR3 records (tools/wex_cuda_bridge.cu, make wex-cuda-helper) and its GPU test
- [ ] #2884 wex: COLI_WEX_CUDA_LIB loads the CUDA helper; WNR3 records decode on the GPU through coli_wex_decode_record

The helper is opened only when `COLI_WEX_CUDA_LIB` names it; unset, every record decodes on the CPU as in series 2. On 16 records of the published V4.1 container, an RTX 3070 decoded a record in 4-9 ms where the CPU reference decoder took about a second (#2884 has the conditions).

Series 4 (I/O switches, opt-in), stacked on series 3:

- [ ] #2885 qwen38, deepseek_v41: a WEX4 container read from COLI_MODEL_MIRROR copies, split by COLI_DISK_WEIGHTS
- [ ] #2886 deepseek_v41: a batch's first-read SHA-256 on the OpenMP team (COLI_WEX_VERIFY)

Series 5 (fast CPU decoder for WNR3), stacked on series 4:

- [ ] #2888 wex: a fast CPU decoder for WNR3 records (wex_fast_decode.h) and its test against the reference
- [ ] #2889 wex: WNR3 records decode on the CPU with the fast decoder (COLI_WEX_FAST_DECODE), AVX2 or NEON chosen at run time

Why: on the i7-7700K, CPU only, DeepSeek V4.1 with the published WNR3 container ran about 3x slower than with the shards (#2879), because the reference decoder (about 761 ms per 18.8 MB expert on one thread) is the bottleneck; Wex-WRN2 measured its fast decoder at 7-8x the reference on one thread, with the same bytes and the same verdicts.

Measured with series 5 (#2889 has the conditions): a V4.1 record 742 -> 101 ms on one i7-7700K thread; DeepSeek V4.1 on the published container (on NVMe), CPU only, 8 tokens decoded in 71.1 s against 293.2 s with the reference, the shards 62.5-91.6 s in the same session, the same tokens. `COLI_WEX_FAST_DECODE=0` keeps the reference. With the CUDA helper the same run takes 35.2 s (0.23 tok/s; every record on the GPU since #2883 pools its workspaces per call: before, it leaked the VRAM of reader threads that exited, a third of the records went back to the CPU and the run took 57.9 s), and with the Vulkan backend (30 of 40 layers on an RTX 3070) plus the helper 48.0 s (was 48.7 s).

Not proposed: splitting one record across two drives (-8.7% decode in the downstream fork), one read per V4.1 record instead of six (+10% on an NVMe drive, -11% on a SATA SSD, through deepseek_v41's reader on one layer's 192 experts), sub-reads and the batched tier prewarm (their gains were measured in a store `dev` does not have).

Series 7 (bench), stacked on series 5:

- [ ] #2894 wex: tools/wex_bench.py, one command for the WEX measurements helpers are asked for

`c/tools/wex_bench.py` runs the measurements of "Tests wanted" below on one machine and writes a report to paste here; `COLI_WEX_STATS=1` (in the same PR) prints which decoder took the WNR3 records. On the i7-7700K it gave the shards' 8 tokens in all four arms (shards, container with the fast decoder, with the reference, with the CUDA helper), each arm's 2,257 records on the decoder it asked for.

Series 6 (GLM-5.3 from a WEX4 container), stacked on series 7:

- [ ] #2895 glm53: AMD MXFP4 checkpoint as published (amd/GLM-5.3-Flash-Quark-MXFP4), experts as fmt 7
- [ ] #2896 glm53: MXFP4 routed experts from a WEX4 container (COLI_WEX_PATH, COLI_WEX_ROOT), RAW and WNR3

glm53 reads AMD's MXFP4 checkpoint as published (routed experts streamed as fmt 7 through `matmul_mxfp4`, the tiers given `VKT_SRC_MXFP4_E8M0`), then takes its routed experts from a WEX4 container written by Wex-WRN2's `pack --names qwen38`, RAW or WNR3, through `expert_read`, the path every MoE path and tier uses. Tested with synthetic shards and containers only (`tests/test_glm53_mxfp4`, `tests/test_glm53_wex4`, Windows and Linux, ASan and UBSan): no whole-model GLM-5.3 run was done; see "Tests wanted". The kit that builds and checks a container of this checkpoint is Wex-WRN2's `testkits/glm53-wex4/`.

## Order and dependencies

```
dev ── #2861 format ── #2862 reader ── #2863 deepseek-v4 ── #2864 qwen38 ── #2865 deepseek_v41
                                                             (the #2825 MXFP4 record layout)
     ── series 2: #2876 wex_decode.c ── #2877 deepseek-v4 ── #2878 qwen38 ── #2879 deepseek_v41 ── #2880 encoder
     ── series 3: #2883 helper ── #2884 loader ── series 4: #2885 mirrors ── #2886 hash on the team
     ── series 5: #2888 fast CPU decoder ── #2889 its use in coli_wex_decode_record
     ── series 7: #2894 the bench (tools/wex_bench.py, COLI_WEX_STATS)
     ── series 6: #2895 glm53 on the MXFP4 checkpoint ── #2896 glm53 from a WEX4 container
```

Each PR is stacked on the one before it and lists the commits to review. The three hooks do not use each other's code; they are stacked only because they add the same variables to `c/coli_env.h` and lines to the same Makefile, docs page (`docs/wex.md`) and CHANGELOG, so merging them in this order gives no conflict. Series 2 extends the hooks and brings `c/wex_decode.c`/`.h` (the WNM2 model registry and record decode it needs; nothing in series 1 calls them).

Test containers built from real checkpoints: [Kaira666/Kaira-WEX4-Experts](https://huggingface.co/Kaira666/Kaira-WEX4-Experts) (Qwen3.8-Flash-Next MXFP4 with RAW records; DeepSeek V4.1 Flash REAP-192 with WNR3 records, for series 2).

## Testing the whole series

Without a model, from `c/`: `make test-c` runs every WEX test (format, reader, `wex_decode.c`, the three hooks on synthetic RAW and WNR3 containers); the Python suite runs the encoder's test. With a model: write a container with Wex-WRN2's `wex4/wex4_convert.py pack` (or `upgrade` of a WEX1 file), check it with `tools/wex4_verify FILE`, then run the engine with `COLI_WEX_PATH=FILE` and the printed `COLI_WEX_ROOT=...`, and compare tokens and logits with the same run without `COLI_WEX_PATH`. Each hook PR gives the exact commands.

## Tests wanted (help welcome)

Every number in this issue was measured on one Windows 10 desktop (i7-7700K, RTX 3070) and its WSL2, plus one Apple M4 run of the decoder test on synthetic records (thanks, mfethe1, on #2888). These are the tests still missing. Each one is a run of `c/tools/wex_bench.py` (#2894; Python 3, standard library only) on the branch of #2894 or any later one, unless the row says otherwise.

**Getting a container.** The published ones are on [Kaira666/Kaira-WEX4-Experts](https://huggingface.co/Kaira666/Kaira-WEX4-Experts): `v41-reap/v4_flash_experts.nr3.wex` (DeepSeek V4.1 Flash REAP-192, 7,680 WNR3 records, 123.3 GB) and `qwen38-mxfp4/q38_experts.wex` (Qwen3.8-Flash-Next, 24,576 RAW records, 64.3 GB). To make your own: a RAW one with Wex-WRN2's `wex4/wex4_convert.py pack`, a WNR3 one with #2880's encoder (`c/tools/build_nr3.py`, then `wex4_convert.py upgrade` to WEX4). The decode, io and verify rows need only the container, not the model; for a quick run, `--records 4` reads 4 records (75 MB).

**Running.** From the repository root, with `make` and a C compiler (the bench builds `tools/wex4_verify` and `tools/wex_decode_bench` itself):

```sh
python3 c/tools/wex_bench.py all --container v4_flash_experts.nr3.wex --out bench-out          # env, verify, decode, io, report
python3 c/tools/wex_bench.py engine report --snap <checkpoint> --container v4_flash_experts.nr3.wex --out bench-out
```

**Reporting.** Paste `bench-out/wex-bench-report.md` as a comment here (it holds the commit, the variables set, the container's root, the records used, and no host name, user name or path; read it before pasting). Say which row it is for, and anything unusual (another job on the drive, a laptop on battery). A failure is as useful as a pass: paste the report and the failing command's output.

Below, `V41` is `v4_flash_experts.nr3.wex`, `Q38` is `q38_experts.wex`, `bench` is `python3 c/tools/wex_bench.py`, and every bench command ends with `--out bench-out`.

### CPU decoders

| Test | What to run | Hardware / OS | Validates | Expected |
|---|---|---|---|---|
| Native ARM64 Linux (NEON) | `make -C c tests/test_wex_fast_decode && c/tests/test_wex_fast_decode --mutate 3000 V41 8`, then `bench env decode io report --container V41` | Any ARM64 Linux: Graviton 2/3/4, Ampere Altra, a Neoverse server, Raspberry Pi 5 | #2888, #2889 | `kernel 1 on NEON here`, 0 mismatches; decode: every arm "same", the fast decoder several times the reference on one thread (M4, synthetic records: 5.4x) |
| SVE, for later | the same, on Graviton 3 or another Neoverse V1 | Graviton 3 / Neoverse V1 (SVE 256-bit) | #2888 (its section 10) | `env` lists SVE; the NEON kernel runs and is identical. The numbers are the baseline for a later SVE kernel (Wex-WRN2's gather kernel is not carried) |
| Apple silicon end to end | `bench env decode io report --container V41`, then `make -C c deepseek_v41` and `bench engine report --snap <V4.1 checkpoint> --container V41` if it builds | Any M1-M5 Mac | #2889, #2879 | decode on real records (the M4 run used synthetic ones): NEON, identical; engine: the shards' tokens in every arm |
| `WEX_FAST_LANES` per CPU | `bench decode report --container V41 --lanes 1,2,3,4,6,8 --records 16` | Any CPU, real records: Zen 2-5, Intel hybrid (12th gen on), Apple, Graviton, Neoverse | #2888, #2889 | identical in every arm; the fastest stream count for that CPU. Measured: i7-7700K 4 (117 ms against 147-161 for 2, 6, 8); M4 6 (synthetic) |
| x86-64 without AVX2 | `bench env decode report --container V41 --records 4` | A pre-Haswell CPU, or an Atom / Celeron without AVX2 | #2889 | `kernel: scalar`; the engine falls back to the reference (only checked under qemu `-cpu Westmere` so far) |

### CUDA helper

| Test | What to run | Hardware / OS | Validates | Expected |
|---|---|---|---|---|
| Linux build and run | `make -C c wex-cuda-test CUDA_ARCH=sm_XX`, then `bench decode report --container V41 --helper c/kaira_wex_cuda.so`, then `bench engine report --snap <V4.1 checkpoint> --container V41 --helper c/kaira_wex_cuda.so` | Linux, an NVIDIA GPU, the CUDA toolkit | #2883, #2884 | `ALL OK` (with the thread-churn VRAM check); decode: "the helper decoded N of N calls", same bytes; engine: the shards' tokens, every record on the helper (`decodes: helper N / fast 0 / reference 0`) |
| Other NVIDIA GPUs and drivers | the same commands (Windows: `make -C c wex-cuda-helper` with MSVC + CUDA, then `--helper c/kaira_wex_cuda.dll`) | Turing (sm_75, below sm_80 never ran), Ada (sm_89), Hopper (sm_90), Blackwell (sm_120), laptop GPUs, older and newer drivers | #2883, #2884 | the same bytes; ms per record (RTX 3070: 6.4 one a call, 4.0 in a host batch of 8) |

### Other GPUs (Vulkan backend)

| Test | What to run | Hardware / OS | Validates | Expected |
|---|---|---|---|---|
| AMD and Intel GPUs | `make -C c deepseek_v41 VK=1`, then `COLI_VULKAN=1 bench engine report --snap <V4.1 checkpoint> --container V41 --arms shards,container-fast,container-reference` | AMD Radeon (RDNA 2-4) or Intel Arc, Linux or Windows | #2879, #2889 | the shards' tokens in every arm; the records decoded by the fast CPU decoder (`decodes: 0 / N / 0 / 0`); tok/s against the shards |

### Whole models and A/B against the shards

| Test | What to run | Hardware / OS | Validates | Expected |
|---|---|---|---|---|
| Qwen3.8, a whole model with WNR3 records | make a WNR3 container of the Qwen3.8 experts (#2880's `build_nr3.py`, then `wex4_convert.py upgrade`), then `bench engine report --engine qwen38 --prompt-ids <ids> --snap <Qwen3.8 checkpoint> --container q38.nr3.wex` | A machine with the Qwen3.8-Flash-Next checkpoint (its FP8 trunk) | #2878, #2889 | the sidecar's tokens in every arm |
| DeepSeek V4, a whole model with WNR3 records | `make -C c deepseek_v4`, then `bench engine report --engine custom --snap <V4 checkpoint> --container <V4 WNR3 container> --cmd "./deepseek_v4 <V4 checkpoint> --prompt-file prompt.txt --max-tokens 8 --no-dspark"` | A machine with a DeepSeek V4 checkpoint (none here) | #2863, #2877, #2889 | the same output in every arm (the bench compares stdout); `[v4_wex] reads=... fallbacks=0 refused=0` in the logs. The `custom` and `qwen38` drivers of the bench have not run yet: a fix is welcome too |
| RAW container against the shards on `dev` | V4.1: pack the RAW container (`wex4_convert.py pack --experts raw`, 144.4 GB), then `bench engine report --snap <V4.1 checkpoint> --container v4_flash_experts.wex --arms shards,container-fast`; Qwen3.8: the same with `--engine qwen38` and `Q38` | V4.1 or Qwen3.8 checkpoint, one NVMe drive | #2864, #2865 | the same tokens; the downstream fork measured +13% decode with the RAW container (16 GB RAM, Windows 11), never measured on `dev` |
| The engram tables | `bench engine report --snap <V4.1 checkpoint with the engram tables> --container V41` | The V4.1 checkpoint with its engram tables (203 GB more) | #2865, #2879 | the shards' tokens in every arm with the model's real output distribution; every run here had engram off |
| GLM-5.3-Flash, a whole model, shards against the container | build glm53 from #2896 (`make -C c glm53`); make the container with Wex-WRN2's kit (`bash testkits/glm53-wex4/run.sh --work <disk> all --colibri <colibri>/c`, which downloads `amd/GLM-5.3-Flash-Quark-MXFP4`, 185 GB, packs it with WNR3 records, checks it and runs the engine stage), or by hand: `./glm53 --model <checkpoint> --ids 785,6722,315,9621,374 --greedy 8`, then the same with `COLI_WEX_PATH=<container> COLI_WEX_ROOT=<root>`, then with `COLI_WEX_CUDA_LIB` too | about 350 GB free on an NVMe SSD, 32 GB of RAM or more; an NVIDIA GPU for the helper arm | #2895, #2896 | the shards' tokens in every arm (`teacher_forcing` and `greedy` lines equal); `[glm53] WEX4 records: N read, M WNR3 decoded (CUDA helper H, CPU C)`, with H = M when the helper is given; tok/s of each arm. Never run: the only machine here has the checkpoint on a SATA SSD |
| GLM-5.3-Flash on the GPU tiers | `make -C c VK=1 glm53` (or the CUDA build), `COLI_VULKAN=1` (or `COLI_CUDA=1`) with and without `COLI_WEX_PATH` | A GPU with enough VRAM for part of the experts | #2895, #2896 | the CPU run's tokens; the tier's line names MXFP4 experts; built here only as `-fsyntax-only` |
| Low-RAM machines (16 GB) | `bench all --snap <V4.1 checkpoint> --container V41` | 16 GB of RAM, the container on NVMe | #2879, #2889 | it completes with the shards' tokens; tok/s of the container against the shards |

### I/O

| Test | What to run | Hardware / OS | Validates | Expected |
|---|---|---|---|---|
| Two NVMe drives: mirrors and split | copy `V41` to a second drive (same file name), `bench io report --container <each copy>`, then `COLI_MODEL_MIRROR=<dir of the copy> bench engine report --snap <V4.1 checkpoint> --container V41 --arms container-fast`, and again with `COLI_DISK_WEIGHTS=1,1` | Two NVMe drives, Linux or Windows; two drives of the same speed especially | #2885 | the same tokens; decode faster than one drive (downstream fork: +49% on a second drive); `tools/wex4_verify` prints the same root for both copies |
| macOS, the container on an external drive | `bench env verify io decode report --container /Volumes/<drive>/v4_flash_experts.nr3.wex` | A Mac with a USB or Thunderbolt SSD | #2862, #2889 | `env` says External; uncached reads (F_NOCACHE) GB/s; verify passes with the published root |
| A CPU with SHA extensions | `bench env verify report --container V41` | x86 with SHA-NI (Zen, Intel 11th gen on), or ARMv8 with SHA2 | #2862, #2886 | `env` lists SHA-NI; verify GB/s against the 1.08 GB/s of this machine without it |

### Builds

| Test | What to run | Hardware / OS | Validates | Expected |
|---|---|---|---|---|
| MSVC builds | build `c/tests/test_wex_fast_decode.c`, `c/tests/test_wex_decode.c` and `c/tools/wex_decode_bench.c` with `cl /O2 /openmp:llvm` or `clang-cl`, run the tests, then put that build at `c/tools/wex_decode_bench.exe` and run `bench decode report --container V41` | Windows, Visual Studio 2022 | #2876, #2888, #2889 | today only gcc and clang build these files (`wex_decode.c` uses the GCC atomics), so a `cl` build may fail: the errors are what we need |

## Open after the series

- A WEX4 writer in colibri (today: the encoder of #2880 writes WEX3, Wex-WRN2's `wex4_convert.py upgrade` makes WEX4).
- Speed numbers measured on colibri `dev` itself, per engine and per backend.
- The Vulkan and Metal paths for WNR3 records (WNR3 on Vulkan exists as a prototype in Wex-WRN2).

