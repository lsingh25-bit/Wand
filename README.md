# The Edge of Magic — TinyML spell-recognition wand

**ESP32-WROOM-32 + BNO055 (IMU fusion mode)**, on a 500 mm shaft. All inference on-chip.

Full rationale, architecture and code walkthrough: `docs/Edge_of_Magic_Technical_Report.pdf`

## The wand is part of the model

    BUTT 0cm ── ESP32 ── power ── OLED @22cm ── BNO055 @37cm ── LED ring @50cm TIP
                                                (+X points at the tip)

Those three numbers live in `shared/wand_config.h` as `WAND_LENGTH_M`,
`SENSOR_FROM_TIP_M` and `SENSOR_FROM_BUTT_M`, and the synthetic bootstrap generator
**derives** the accelerometer signal from them via rigid-body kinematics
(`a = a_pivot + α×r + ω×(ω×r)`). A sensor 37 cm up a rigid shaft does not measure the
caster's hand; it measures the hand plus a lever arm, and the lever term is the larger by
three to ten times.

**If you build a different wand, change those three constants — not the generator — then
regenerate, re-audit, and re-sweep.** Everything downstream follows automatically, because
nothing downstream re-types them.

## Quick start (no hardware, no data required)

    pip install tensorflow-cpu numpy pytest pyserial
    ./tools/hostcheck/check.sh                          # firmware compiles, no ESP32 toolchain
    python training/synth_bootstrap.py                  # quarantined bootstrap corpus
    python training/audit_corpus.py data/synthetic      # is it physically plausible?
    python -m pytest training/test_config_parity.py -v
    cd training && python parity_check.py --data ../data/synthetic --n 60 && cd ..
    python training/train_cnn.py --data data/synthetic --allow-synthetic --epochs 60
    python docs/build_report.py

`data/synthetic` is a QUARANTINED bootstrap corpus. No number derived from it may enter
the report, the PPT, or any submitted result. Better physics makes it a better engineering
target — the tensor ranges, the saturation rate and the gate pass rates are now the ones
the real wand will produce — and does **not** make it evidence. The loader refuses
`data/raw` until it holds at least 50 genuine captures.

## Once real gestures exist — in this order

    python training/collect.py --port COM5 --caster <name> --gesture LUMOS --reps 15
    python training/audit_corpus.py data/raw          # BEFORE collecting the other 475
    cd training && python parity_check.py --data ../data/raw --n 30 && cd ..
    python training/train_cnn.py --data data/raw
    python training/calibrate.py --data data/raw --misfire-budget 0.0   # writes wand_thresholds.h

Run `audit_corpus.py` on the first fifty captures and stop if anything is outside its band.
Those bands are what the tensor divisors, the Stage 1 gate and the arena were sized against;
a mismatch means one of them is describing a different wand, and finding that out after 525
captures is expensive.

`calibrate.py` is not optional. The three softmax thresholds are measurements, not design
choices; shipping the placeholders costs roughly half the recall (report §8.4).

## Firmware — three targets, one driver

    pio run -e bringup  -t upload -t monitor    # FIRST: 6-test acceptance suite
    pio run -e logger   -t upload -t monitor    # collection, serial ON
    pio run -e infer_nn -t upload               # DEMO BUILD, serial not compiled

`firmware/common/bno055_fusion.cpp` is the only BNO055 driver and all three link it, so the
bring-up suite tests exactly the configuration that ships.

**Run `-e bringup` before soldering anything.** Test 5 checks the mounting geometry (point
the tip at the floor; body-frame "up" must read ≈ (−1, 0, 0)) and test 6 is a ten-minute
clock-stretching stress test — this part hangs the I2C bus in a way that passes every short
test and dies during the demo.

`DEBUG_SERIAL=0` in the demo target is the defence against the offloading-disqualification
rule — the call is not in the binary.

## Sensor: BNO055, not MPU-6050

Address `0x28`, CHIP_ID at `0x00` reads `0xA0`, `OPR_MODE = 0x08` (IMU fusion, magnetometer
off), bus 0 at **100 kHz** with a 1 s timeout because this part stretches the clock.
The fusion supplies gravity-free acceleration, bias-compensated rate and a quaternion, so
the software gravity/bias/complementary-filter stages are deleted. See report §6.1.

One correction worth knowing: the ±4 g fusion lock applies to the **raw** signal, and the
fusion subtracts gravity afterwards, so linear acceleration is reportable to **±5 g per
axis** — the rail plus one gravity, not minus it. `NORM_ACC_LIN` is 5.0 for that reason
(report §6.5).
