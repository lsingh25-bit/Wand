#!/usr/bin/env bash
# tools/hostcheck/check.sh -- type-check every firmware translation unit on a
# laptop, with no ESP32 toolchain and no hardware.
#
# This does NOT run the firmware and it is NOT a substitute for `pio run`. What
# it catches is the class of mistake that otherwise costs a full flash cycle to
# find: a typo, a missing brace, a function renamed in a header but not at its
# call site, a constant that moved between headers. On a project where the
# toolchain lives on one teammate's laptop, that is most of the round trips.
#
# TFLite Micro is stubbed too, with the exact signatures wand_infer.cpp calls, so
# a call that no longer matches the library fails here instead of on the ESP32
# toolchain. The stubs compute nothing -- this proves the code COMPILES, never
# that it works.
set -u
cd "$(dirname "$0")/../.."
STUBS=tools/hostcheck/stubs
INC="-I$STUBS -Ishared -Ifirmware/common -Ifirmware/wand_infer"
FLAGS="-fsyntax-only -std=gnu++17 -Wall -Wextra -Wno-unused-parameter -DDEBUG_SERIAL=1"
fail=0
check() {
  printf '  %-46s' "$1"
  if out=$(g++ $FLAGS $INC $2 "$1" 2>&1); then echo "OK"
  else echo "FAIL"; echo "$out" | head -25; fail=1; fi
}
echo "host type-check (stubbed Arduino; not a functional test)"
check firmware/common/bno055_fusion.cpp   ""
check firmware/wand_infer/preprocess.cpp  ""
check firmware/wand_infer/reject.cpp      ""
check firmware/wand_infer/parity_host.cpp ""
check firmware/wand_logger/wand_logger.cpp ""
check firmware/wand_tools/bno_bringup.cpp  ""
check firmware/wand_infer/wand_infer.cpp  ""
exit $fail
