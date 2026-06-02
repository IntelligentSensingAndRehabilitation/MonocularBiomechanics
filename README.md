<div align="center">

# Portable Biomechanics Laboratory: Clinically Accessible Movement Analysis from a Handheld Smartphone

[J.D. Peiffer](https://www.sralab.org/researchers/jd-peiffer)<sup>1,2</sup>, Kunal Shah<sup>1</sup>, Irina Djuraskovic<sup>1,3</sup>, Shawana Anarwala<sup>1</sup>, Kayan Abdou<sup>1</sup>, Rujvee Patel<sup>4</sup>, Prakash Jayabalan<sup>1,5</sup>, Brenton Pennicooke<sup>4</sup>, R. James Cotton<sup>1,5</sup>

<sup>1</sup>Shirley Ryan AbilityLab, Chicago, IL<br>
<sup>2</sup>Biomedical Engineering, Northwestern University, Evanston, IL<br>
<sup>3</sup>Interdepartmental Neuroscience, Northwestern University, Chicago, IL<br>
<sup>4</sup>Neurological Surgery, Washington University School of Medicine, St. Louis, MO, USA<br>
<sup>5</sup>Physical Medicine and Rehabilitation, Northwestern University Feinberg School of Medicine, Chicago, IL, USA<br>

</div>
<img src="docs/static/images/overlay_fig.jpg" width="800">

> This repository includes code and a gradio demo for running the single camera (monocular) biomechanical fitting code from smartphone videos.

# Abstract
Movement directly reflects neurological and musculoskeletal health, yet objective biomechanical assessment is rarely available in routine care. We introduce Portable Biomechanics Laboratory (PBL), a platform for fitting biomechanical models to handheld smartphone video. We validate PBL on over 15 hours of data synchronized to ground truth motion capture, finding joint-angle errors  < 3 degrees across patients with neurological injury, lower-limb prosthesis users, pediatric inpatients, and controls. Across 1021 videos recorded in prospective clinical deployment, PBL was easy to implement, yielded reliable gait metrics (ICC > 0.9), and detected clinically relevant differences in movement. For cervical myelopathy patients, its gait quality measures correlated with modified Japanese Orthopedic Association (mJOA) scores and were responsive to clinical intervention. Handheld smartphone video can therefore deliver accurate, scalable, and low-burden biomechanical measurement, enabling greatly increased monitoring of movement impairments. We release the first clinically validated method for measuring whole-body kinematics from handheld smartphone video at [https://github.com/IntelligentSensingAndRehabilitation/MonocularBiomechanics](https://github.com/IntelligentSensingAndRehabilitation/MonocularBiomechanics).

<video src="docs/static/videos/jd_running.mp4" width="800" controls autoplay muted loop></video>

# Code
Tested with Ubuntu 24.04.3 LTS using [uv](https://docs.astral.sh/uv/).

Clone and install
```
git clone git@github.com:IntelligentSensingAndRehabilitation/MonocularBiomechanics.git
cd MonocularBiomechanics/
uv sync --extra cuda  # GPU (CUDA 12)
# uv sync            # CPU only
```

**EGL/OpenGL system dependencies:** The rendering pipeline uses EGL (via PyOpenGL and MuJoCo). On Ubuntu/Debian, install the required system packages before running:
```
sudo apt-get install libgl1 libegl1 libgles2 libosmesa6-dev libglib2.0-0 ffmpeg
```
`MUJOCO_GL` and `PYOPENGL_PLATFORM` default to `egl`. Update these environment variables if you need to use a different rendering backend.

Note: Setting `"setuptools==81.0.0"` in pyproject.toml may help resolve some tensorflow version errors. 

Note 2: Windows is not supported. WSL may work for biomechanical fitting but is not supported for overlay creation.


## Gradio demo
```
uv run python main.py
```
A local webpage will open to upload and run the code.

Note that this demo is not optimized for videos with many people in view -- if you want to do so, consider using [PosePipeline](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline) to annotate the person of interest.

**This demo is intended for proof-of-concept fitting only.** Production runs should consider using [PosePipeline](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline) or contact the authors for a Docker container.

**Camera intrinsics:** The demo uses default camera intrinsics from a Samsung Galaxy S20 (see [`monocular_demos/dataset.py:get_samsung_calibration`](monocular_demos/dataset.py#L510)). Most proof-of-concept fits will work fine with these defaults, but results may vary if your camera's intrinsics differ significantly. For best results, provide calibration parameters specific to your device.


# Jupyter Notebook
A jupyter notebook with steps to run the pipeline can be found [here](https://github.com/IntelligentSensingAndRehabilitation/MonocularBiomechanics/blob/main/monocular-demo.ipynb).

# Reproducing Paper Analysis
To reproduce the error tables from the paper, first download the data from [Zenodo](https://doi.org/10.5281/zenodo.20450570) and place the CSV files in `data/clinical/`, then run:
```
python analysis/compute_clinical_errors.py
```

# Citation
This work has been presented at the [2024 American Society of Biomechanics Meeting](https://drive.google.com/open?id=1CEZBhwAYALvUds0VbFy50U1LmOfgS0kO&usp=drive_fs) and [2025 European Society of Biomechanics Meeting](https://drive.google.com/open?id=19y1_F-0o5CVRFdihe-0kReQ9baH-jFX4&usp=drive_fs).


```bibtex
@misc{peiffer_portable_2025,
	title = {Portable Biomechanics Laboratory: Clinically Accessible Movement Analysis from a Handheld Smartphone},
	doi = {10.48550/arXiv.2507.08268},
	number = {{arXiv}:2507.08268},
	publisher = {{arXiv}},
	author = {Peiffer, J. D. and Shah, Kunal and Djuraskovic, Irina and Anarwala, Shawana and Abdou, Kayan and Patel, Rujvee and Jayabalan, Prakash and Pennicooke, Brenton and Cotton, R. James},
	date = {2025-07-11},
}
```

