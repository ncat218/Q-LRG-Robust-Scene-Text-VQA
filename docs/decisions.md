# Decisions

- Sample images by SHA-256 rank using seed 42. Reserve 10% of official train for val_inner and retain every question for each diagnostic image. Increasing the limit extends the same prefix. Official validation remains available for later reporting.
- Decode clean RGB in the raw annotation orientation. Verify the raw TextOCR dimensions and record EXIF orientation. Apply Pillow 11.3.0 Bicubic in memory to both resize steps.
- Project polygons through the processor's full-canvas resize into the row-major merged grid. The default mask accepts at least 5% polygon coverage of a cell. Record token counts under the alternate center rule. Run a separate S0 directory with a modified S0 config for a paired accuracy sensitivity comparison.
- For O4, randomly order possible translations and prefer a disjoint location. When every translation overlaps, use the minimum overlap found. If there is no different translation, sample the same number of tokens, preferring tokens outside O3. Record the mode and overlap for each question.
- Write atomic per-image prediction shards. Resume only if the manifest, code, configuration, environment and prior D0 binding match. A completed run includes artifact checksums.
- The full launcher requests one 35 GB MIG slice, four CPUs, 32 GB RAM and a three-hour walltime limit. Full submission is selected with `--mode full`. Separate stage commands support explicit resource settings.
