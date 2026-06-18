# 3D Scene Reconstruction Assignment Pipeline

This project prepares the non-report part of the assignment: data preprocessing,
scene representation training adapters, evaluation, timing, and result
collection. The report PDF is intentionally not generated here.

The project reuses the existing Conda environment `torch_env`; it does not
create a project-local `.venv`.

## 1. Environment

Use the main environment:

```powershell
conda activate torch_env
```

Or run commands directly:

```powershell
conda run -n torch_env python scripts/check_env.py --dry-run
```

Current configured tools are in `config/project.yaml`:

- Python: `G:/anaconda/envs/torch_env/python.exe`
- COLMAP: `E:/3D-tools/COLMAP/bin/colmap.exe`
- ffmpeg: not configured yet; set `env.ffmpeg` or add it to PATH

Environment report:

```powershell
conda run -n torch_env python scripts/check_env.py --dry-run
```

Output:

```text
logs/env_report.json
```

## 2. Directory Layout

```text
data/
  raw/desk.mp4
  images/
  splits/train.txt
  splits/test.txt
  colmap/
results/
  colmap/
  nerfstudio/
  3dgs/
  vggt_gs/
  metrics.csv
  timing.csv
logs/
external/
```

Place third-party repositories under `external/` when needed:

```text
external/gaussian-splatting/
external/vggt-omega/
```

## 3. Dry-Run Before Video Exists

These commands should be safe before recording the video:

```powershell
conda run -n torch_env python scripts/inspect_video_camera.py --dry-run
conda run -n torch_env python scripts/prepare_video.py --dry-run
conda run -n torch_env python scripts/split_dataset.py --dry-run
conda run -n torch_env python scripts/run_colmap.py --dry-run
conda run -n torch_env python scripts/run_nerfstudio.py --dry-run
conda run -n torch_env python scripts/run_3dgs.py --dry-run
conda run -n torch_env python scripts/run_vggt_gs.py --dry-run
conda run -n torch_env python scripts/collect_results.py --dry-run
conda run -n torch_env python scripts/evaluate_all.py --dry-run
```

They preview commands and paths. Missing external dependencies are reported
instead of hidden.

Current orchestration helper:

```powershell
conda run -n torch_env python scripts/pipeline.py status
conda run -n torch_env python scripts/pipeline.py prepare-all --dry-run
conda run -n torch_env python scripts/run_experiment_suite.py --method all --stage all --dry-run
```

## 4. Full Workflow After Recording

The compact route is:

```powershell
python scripts/pipeline.py prepare-all
```

This runs frame extraction, split generation, all-view COLMAP, train/test
physical split, train-only COLMAP, COLMAP geometry analysis, 3DGS data
preparation, and method dry-runs.

The explicit route is:

1. Put the raw video at `data/raw/desk.mp4`, or update `data.video` in
   `config/project.yaml`.
2. Extract frames:

```powershell
python scripts/inspect_video_camera.py
python scripts/prepare_video.py
```

3. Create shared train/test split:

```powershell
python scripts/split_dataset.py
```

4. Run COLMAP sparse reconstruction:

```powershell
python scripts/run_colmap.py
```

5. Convert COLMAP poses to Nerfstudio transforms and prepare 3DGS source data:

```powershell
python scripts/colmap_to_nerfstudio.py
python scripts/prepare_3dgs_data.py
```

Geometry quality artifacts from train-only COLMAP:

```powershell
python scripts/analyze_colmap_geometry.py
```

Output:

```text
results/colmap/train_geometry_summary.json
results/colmap/train_camera_centers.csv
results/colmap/train_geometry_preview.png
```

Camera/video audit:

```text
logs/video_camera_report.json
results/camera_parameters.md
```

6. Run Nerfacto/NeRF via nerfstudio:

```powershell
python scripts/run_nerfstudio.py
```

After training, run native Nerfstudio evaluation and optional render export:

```powershell
python scripts/run_nerfstudio.py --mode eval
```

7. Run official 3DGS adapter:

```powershell
python scripts/run_3dgs.py
```

8. Run VGGT-to-COLMAP and GS adapter:

```powershell
python scripts/run_vggt_gs.py
```

9. Evaluate rendered hold-out images:

```powershell
python scripts/evaluate_render.py --method 3dgs --pred-dir results/3dgs/eval_model/test/ours_30000/renders --gt-dir results/3dgs/eval_model/test/ours_30000/gt
python scripts/evaluate_all.py --dry-run
python scripts/evaluate_all.py
```

10. Summarize outputs:

```powershell
python scripts/collect_results.py
```

## 5. Timing Wrapper

Wrap long commands with:

```powershell
python scripts/timed_run.py --method 3dgs --stage train -- python scripts/run_3dgs.py --mode train
```

This appends timing/resource metadata to `results/timing.csv` and writes a JSON
log under `logs/`.

Unified experiment runner:

```powershell
python scripts/run_experiment_suite.py --method nerfstudio --stage train --env-mode fallback
python scripts/run_experiment_suite.py --method 3dgs --stage render --env-mode fallback
python scripts/run_experiment_suite.py --stage collect
```

## 6. Method Policy

- Representation 1: COLMAP sparse point cloud / geometry.
- Representation 2: nerfstudio Nerfacto/NeRF.
- Representation 3: official 3D Gaussian Splatting.
- Extra comparison: VGGT predicts camera/point/COLMAP-format data, then feeds a
  Gaussian splatting training path.

All methods must use the same extracted images and the same hold-out split.
Do not treat COLMAP or VGGT output as absolute ground truth.

External installation notes are in `INSTALL_EXTERNAL.md`.
The compact command checklist is in `EXPERIMENT_COMMANDS.md`.
The method manifest used by the report is `results/method_manifest.json`, with
the generated readable version in `METHOD_ARCHITECTURES.md`.

Readiness check:

```powershell
conda run -n torch_env python scripts/check_readiness.py
```
