# wand_logger

Data-collection firmware. Serial enabled (`DEBUG_SERIAL=1`), never flashed after collection ends.

    pio run -e logger -t upload
    python training/collect.py --port /dev/ttyUSB0 --caster ravi --gesture LUMOS --reps 15

## What it shares with the demo firmware, and why that is a requirement

The model is trained on what this program writes to the serial port and runs on what
`wand_infer.cpp` reads out of its ring buffer. Any difference between the two is a
distribution shift that no downstream test can see. So both link the same driver
(`firmware/common/bno055_fusion.cpp`), use the same sampler structure, and include
`shared/wand_config.h` without re-typing a single constant.

## The two contracts

**The CSV going out.** `training/collect.py` parses exactly this and `parity_host` reads the
same rows back:

    # GESTURE_START label=<n> caster=<s> rep=<n> nsamples=<n> calib=<0-3> mode=0x08
    idx,qw,qx,qy,qz,lax,lay,laz,gx,gy,gz,dt_us,dup
    ...
    # GESTURE_END

`calib` is the CALIB_STAT gyro field at capture time. It is recorded even though the firmware
refuses to arm below 3, so a marginal capture can be found and discarded later rather than
silently poisoning the dataset.

**The commands coming in.** `SET label=<0-5> caster=<name>`, which is what `collect.py`
already sends. A mismatch here does not fail — it files every capture under the wrong label,
and the first symptom is a confusion matrix nobody can explain.

## Collecting NOISE

Hold the trigger down *through* the behaviour, exactly as for a spell. Leaving the logger
running and calling whatever it records "noise" produces nothing usable: the trigger is what
defines a window, and the false positives that matter are the ones where a judge did press
the button and then did something that was not a spell.

Work through the problem statement's own list — pacing, scratching, gesticulating, setting
the wand down, knocking it on the table — plus the two that matter more than any of them:
half-completed spells, and casts begun and abandoned.

## Before you collect anything

1. Run `pio run -e bringup` first. All six tests. Test 5 (mounting geometry) and test 6
   (ten-minute stress) are the two that are expensive to skip.
2. Check that `FastLED.show()` and every OLED `display()` call stays outside the capture
   window. 240 µs of disabled interrupts mid-capture corrupts the 10 ms grid and the CSV
   still looks perfectly valid. `collect.py` catches this by checking every `dt_us`.
3. Run `python training/audit_corpus.py data/raw` on the first fifty captures and stop if
   anything falls outside its plausibility band.
