# Data layout

Grid36 retains all 960 training scenario configurations, 30 evaluation scenarios and 6 calibration scenarios. Training routes are deduplicated by SHA256: 720 distinct files serve the 960 configurations. Route content and scenario order are unchanged; use route paths from the SUMO configurations rather than deriving route filenames from episode numbers.

Kunshan retains the inputs for training, evaluation, calibration at 100/300/500 metres and published baselines. Unused 1000-metre detectors, unused full-scale observation detectors and historical Grid36 detector/signal variants were removed. The active calibration configurations cover 100/300/500 metres.

Kunshan E1 detector output is written to `results/kunshan_e1_output.xml`, not this input directory. As before, simulations using this fixed detector-output filename should run serially.

The consolidated generator `scripts/generate_grid36.py` writes to `results/generated/grid36` by default and automatically deduplicates training routes. It preserves D50/D70 interpolation and the evaluation emission classes from the distributed inputs. `scripts/generate_kunshan.py` supports base and enhanced demand, writing to `results/generated/kunshan`. Generated routes were compared with the distributed routes as parsed XML, ignoring formatting only. Run `python scripts/check_project.py --update-manifest --hashes` after intentionally changing distributed assets.
