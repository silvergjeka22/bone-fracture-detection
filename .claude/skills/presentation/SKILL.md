---
name: presentation
description: Build, update or extend the LaTeX beamer presentation of the bone-fracture project (presentation/main.tex) - rebuild it after a new run, add a slide, a figure, a table or a number, fix a layout overflow, adjust it to the 12-minute limit, or update the speaker notes. Use it whenever the user mentions the slides, the presentation, the talk, the oral exam presentation, beamer or main.pdf.
---

# Presentation

`presentation/main.tex` (beamer, metropolis, 16:9) never contains a result typed by hand:

| source | produced by | used as |
|---|---|---|
| `results/latex/numbers.tex` | `report.latex_macros` (notebook section 10) | macros such as `\TestAccCNN`, `\BestModel`, `\DupTest` |
| `results/latex/<table>.tex` | `report.export_latex` | `\resulttable[width]{cv}` |
| `results/figures/<name>.png` | `plots.*(save_to=...)` | `\resultfig[opts]{learning_curves}` |
| `presentation/defaults.tex` | by hand | `--` fallbacks so the deck compiles before a run |
| `presentation/figures/` | static images | wavelets (no training needed), v1 figures from the previous iteration |

Missing inputs render as a grey placeholder box, so the PDF always builds.

## Build

```bash
make -C presentation                       # uses ../results
make -C presentation RESULTS=../out/results
./run.sh slides                            # same as the first line
```

Check the output: `pdftoppm -r 60 -png presentation/main.pdf /tmp/s` and look at the pages
(overflows are easy to miss in the log). `grep Overfull presentation/main.log`.

## Adding things

- **A number**: add it in `src/report.py::latex_macros` (`add("Name", value)`), add a
  `\providecommand{\Name}{\na}` fallback in `defaults.tex`, extend `tests/test_report.py`.
- **A figure**: save it from the notebook with `save_to=FIG_DIR / "<name>.png"`, then
  `\resultfig{<name>}` in a frame.
- **A table**: add it to the dict passed to `report.export_latex` in the notebook, then `\resulttable{<name>}`.
- Colours must match the figures: `cnn` (blue), `scat` (orange), `res` (aqua), `head` (violet).

## Time budget

The exam limit is 12 minutes (15 for a pair). The deck has ~17 main slides + backups; the
timing is in `presentation/SPEAKER_NOTES.md` (~40 s per slide). If it runs long, cut slide 18
(agreement) and merge 13-14 (XAI grids) before cutting content the exam asks for (filters,
CV mean accuracy/F1, test >= 75%, XAI on both models, scratch vs Captum, discussion).
After a new run, re-read slides 10 and 19: their reasoning must match the new numbers.
