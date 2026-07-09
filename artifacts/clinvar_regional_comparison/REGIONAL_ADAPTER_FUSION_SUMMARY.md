# Regional Adapter Fusion Summary

All training and regional evaluation jobs completed.

Key test-split findings:

- Brazilian slices improve strongly with M5/M6 versus M0.
- ABRAOM-common benign specificity improves strongly with M5/M6, reducing false pathogenic calls.
- ABRAOM-present pathogenic recall drops sharply with M5/M6, indicating over-suppression by frequency evidence.
- M4 is more conservative: smaller Brazilian gain, less ABRAOM benign suppression, and better pathogenic recall than M5/M6, but weaker overall regional gains.
- Current best scientific read: ABRAOM frequency is useful, but the current frequency weighting is too strong for pathogenic variants and needs sensitivity protection/calibration before clinical use.

Artifacts:

- `m0_m4_m5_m6_regional_test_summary.csv`
