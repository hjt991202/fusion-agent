# Demonstration artifacts

- Included: docs/assets/demo-comparison.png, generated from actual CPU outputs on original
  synthetic visible/infrared inputs. Visually inspected: four panels have matching shape;
  Haar retains sharper outlines while weighted fusion smooths edges.
- To capture: Gradio initial screen with both uploaded synthetic images and task text.
- To capture: completed gallery, four-metric comparison, expanded tool trace and report download.
- To record: short GIF showing upload, run and report export, with no keys or private files visible.
- Optional: missing-input error and explicit no-key fallback demonstration.

Do not add nonexistent GIF/screenshot links to README. The existing comparison figure is an
algorithm output montage, not a screenshot of the web UI. Local server startup was checked via HTTP.

Reproduction: run examples/generate_demo.py, then the CLI with --method both. Use the printed
run folder's weighted.png and wavelet.png alongside the two inputs for the comparison.
