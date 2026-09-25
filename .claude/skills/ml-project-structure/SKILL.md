---
name: ml-project-structure
description: Restructure or scaffold a machine-learning / deep-learning / data-science notebook project into a simple layout where ALL code lives in flat src/ modules (data, models, training, plots, utils, plus e.g. xai) and one thin notebook only sets parameters and calls those functions, with results/ for outputs and archive/ for legacy material. Use this whenever the user wants to clean up, organize, refactor or simplify a Jupyter/Colab ML project, move notebook code into src, "make the notebook simpler", remove duplicated or dead functions, prepare a course/exam/thesis project for submission, or start a new ML project with a clean structure, even if they don't say "structure" explicitly.
---

# ML project structure: code in `src/`, thin notebook

The goal is a project a reader (examiner, teammate, future you) understands in five minutes:
the notebook reads like the report of the experiment, and every piece of logic has exactly
one home in `src/`. Simplicity is the point. Resist adding layers (config frameworks, class
hierarchies, CLI tools) the project does not need.

## Target layout

```
project/
├── README.md             what it is, how to run it, results, known issues
├── requirements.txt
├── notebooks/
│   └── main.ipynb        the only notebook: settings + calls to src, no function definitions
├── src/
│   ├── __init__.py       docstring listing the modules in pipeline order
│   ├── data.py           dataset class, transforms, loaders, dataset summary
│   ├── models.py         model classes + build_model(name) factory
│   ├── training.py       train loop, (k-fold) CV, test metrics, save/load weights and results
│   ├── plots.py          every figure: one function = one figure, optional save_to=
│   ├── utils.py          seed, device, small helpers
│   └── <topic>.py        one flat module per extra concern (e.g. xai.py), only if needed
├── results/              everything the code produces: models/, cv/, figures/, cached outputs
└── archive/              old notebooks and replaced code, kept until the user deletes it
```

Why flat modules instead of packages with one file per function: a student project has
5 to 8 concerns, and `from src import data, models, training, plots` followed by
`training.train_kfold_cv(...)` reads naturally in a notebook. Deep packages add import noise
and hide duplicates. Split a module only when it grows past roughly 300 lines or holds a piece
the reader must find on its own (e.g. `occlusion_scratch.py` for an "implement it from
scratch" requirement).

## Rules for the notebook

1. **Setup cell**: get the code (on Kaggle: clone the repo; locally: the repo root), `sys.path.insert`,
   then one call to `src/bootstrap.py::setup()` that installs missing packages, finds the data and makes
   the output folders (see this project's `kaggle-run` skill for the Kaggle GPU pattern).
2. **Settings cell**: every tunable in one place (paths, model list, epochs, `TRAIN = False`
   flag, method parameters as a dict). It is the only cell a user should need to edit.
3. **Then one section per step of the pipeline**: a markdown cell saying *what and why*, then
   code cells of about 1 to 8 lines that call `src` functions. No `def`/`class` in the notebook.
4. **Expensive steps are skippable**: `TRAIN = False` loads saved weights and results;
   slow computations (e.g. attributions) take a `cache=` path.
5. **Last expression displays**: functions return DataFrames for tables and plot functions
   draw and show their figure, so cells stay one-liners.
6. **Compare first, then go deep.** When several models are trained, compare them before
   analysing any of them (CV + test table, where each fails), choose the best with its
   cross-validation score (never the test set, which must stay an unbiased estimate), and spend
   the expensive analysis (XAI, error inspection) on that model, plus any model the assignment
   explicitly requires.
7. **End with a discussion section** in markdown (results table, interpretation, limitations).
8. **Commit the notebook without huge outputs**. Figures belong in `results/figures/`.

## Workflow A: restructure an existing project

1. **Study first, change later.** Read every source file and every notebook (extract code
   and text outputs with a short `nbformat` / `json` script; big notebooks are mostly images).
   Record the actual results (metrics, confusion matrices, printed logs) before touching
   anything, because the refactor must not lose them. Note:
   - which functions the notebook really calls (the rest is dead code);
   - duplicates (same function in several modules, or copy-pasted into the notebook);
   - bugs seen in the outputs (wrong labels, a result computed with the wrong model, hard-coded
     device or paths, leaked secrets such as API keys).
2. **Design the module map** (old file/function -> new module) and say it to the user in a
   few lines before moving files.
3. **Move, don't delete.** `git mv` old notebooks and replaced code into `archive/`; git
   history keeps everything and the user decides when to delete `archive/`. Redact secrets in
   anything you keep and tell the user to revoke them (they stay in git history).
4. **Rewrite `src/`** module by module:
   - keep public class names and layer/attribute names so saved weights still load;
   - one implementation per function; merge the variants, keep the one the results used;
   - no hidden globals: functions take explicit arguments with sensible defaults;
   - fix bugs found in step 1 and list every behaviour change for the user.
5. **Build the notebook** with a small `nbformat` script (clean JSON, no outputs), following
   the rules above. If old results exist only as notebook logs, parse them into `results/`
   (e.g. `results/cv/<model>.json`) so the new notebook can show them with `TRAIN = False`.
6. **Update README.md**: layout, how to run (local and Colab), results, known issues.
7. **Verify** (next section) and report.

## Workflow B: new project

Run the scaffold script, then fill the stubs together with the user:

```bash
python .claude/skills/ml-project-structure/scripts/scaffold.py <project_dir> \
    --title "My project" --modules data models training plots utils \
    --sections "Data" "Models" "Training" "Evaluation" "Discussion"
```

It never overwrites existing files, so it is also safe for adding missing pieces to a project.

## Verify before reporting

- `python .claude/skills/ml-project-structure/scripts/check_project.py <project_dir>` checks
  that the notebook defines no functions, cells are short, there are no absolute paths or
  secret-looking strings, and no function name is defined twice across `src/`.
- Import every module (`python -c "from src import data, models, ..."`).
- Unit-check what can be checked without the real data: model forward pass shapes,
  a from-scratch method vs its library version, metrics on a toy example.
- Smoke-run the whole notebook on a tiny synthetic dataset (a few random images per class in a
  temporary folder, 1 epoch, 2 folds, small method parameters) by executing a copy whose
  settings cell is patched, e.g. with `nbclient`. That is the proof the notebook runs top to bottom.
- If something could not be verified (no GPU, no dataset, a missing package), say so plainly.

## Report to the user

Show: the new tree, the old -> new mapping, why each choice (short), bugs fixed and behaviour
changes, the results found in the old outputs, what was verified and how, and open issues
that need their decision. See `references/example.md` for a real before/after.
