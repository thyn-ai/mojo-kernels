#!/usr/bin/env bash
# The Linux x86-64 kernels must run on every x86-64 CPU the release claims.
#
# kernels/*/build.sh compile the shipped kernels with `--target-cpu x86-64-v3`
# (AVX2, FMA, BMI2: every x86-64 CPU since Intel Haswell (2013) and AMD Zen
# (2017), and every GitHub-hosted runner). Without that flag `mojo build`
# targets the CPU of the build machine, and a kernel built on an AVX-512
# runner dies with SIGILL ("Illegal instruction") on any CPU without AVX-512.
#
# This script disassembles each given shared library and refuses any
# EVEX-encoded instruction. In 64-bit mode the byte 0x62 is only ever the
# EVEX prefix (AVX-512, including its 128/256-bit VL forms), so an instruction
# whose encoding starts with 0x62 is one the x86-64-v3 baseline cannot run.
#
# Usage: scripts/check_x86_64_baseline.sh <lib.so> [<lib.so> ...]
# Linux x86_64 only (a build.sh calls it right after `mojo build`); it needs
# `objdump` from binutils on PATH.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <lib.so> [<lib.so> ...]" >&2
  exit 2
fi
if [ "$(uname -sm)" != "Linux x86_64" ]; then
  echo "::error::$0 checks Linux x86_64 libraries; this is $(uname -sm)." >&2
  exit 2
fi
if ! command -v objdump >/dev/null 2>&1; then
  echo "::error::objdump (binutils) is required to check the x86-64-v3 baseline of the built kernels." >&2
  exit 2
fi

status=0
for lib in "$@"; do
  if [ ! -f "$lib" ]; then
    echo "::error::$lib: no such file" >&2
    status=1
    continue
  fi
  # A non-zero objdump exit (not an ELF, unreadable, unsupported format) must
  # name the library and fail closed, and the remaining libraries still get
  # checked. objdump's own diagnostics are kept for the log.
  if ! listing="$(objdump -d "$lib" 2>&1)"; then
    echo "::error::$lib: objdump -d failed; the x86-64-v3 baseline of this library cannot be verified." >&2
    printf '%s\n' "$listing" | tail -n 5 >&2
    status=1
    continue
  fi
  total="$(printf '%s\n' "$listing" | grep -cE '^ +[0-9a-f]+:' || true)"
  if [ "$total" -eq 0 ]; then
    echo "::error::$lib: objdump found no instructions; is this an x86-64 ELF shared library?" >&2
    status=1
    continue
  fi
  # `<addr>:<tab>62 ...` -- the first encoded byte is the EVEX prefix.
  evex="$(printf '%s\n' "$listing" | grep -cE $'^ +[0-9a-f]+:\t62 ' || true)"
  if [ "$evex" -ne 0 ]; then
    echo "::error::$lib: $evex of $total instructions are EVEX-encoded (AVX-512); the kernel was not built for the x86-64-v3 baseline and will SIGILL on CPUs without AVX-512. Build it with \`mojo build --target-cpu x86-64-v3\` (see kernels/*/build.sh)." >&2
    # awk reads the whole listing (no early exit, so no SIGPIPE under pipefail).
    sample="$(printf '%s\n' "$listing" | awk '/^[0-9a-f]+ <.*>:/ {fn=$2} /^ +[0-9a-f]+:\t62 / {if (n++ < 5) print "  " fn " " $0}')"
    printf '%s\n' "$sample" >&2
    status=1
    continue
  fi
  echo "x86-64-v3 baseline OK: $lib ($total instructions, no EVEX/AVX-512 encoding)"
done
exit "$status"
