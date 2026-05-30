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
Tested with Ubuntu/WSL 2 and NVIDIA T500 laptop GPU.

Clone and install
```
git clone git@github.com:IntelligentSensingAndRehabilitation/MonocularBiomechanics.git
cd MonocularBiomechanics/
pip install -e .
```
Note: For cpu support, modify the pyproject.toml to remove extras from `"jax[cuda12]"` and `"tensorflow[and-cuda]"`.
Note 2: Setting `"setuptools==81.0.0"` may help resolve some tensorflow version errors. 


## Gradio demo
```
python main.py
```
A local webpage will open to upload and run the code.

Note that this demo is not optimized for videos with many people in view -- if you want to do so, consider using [PosePipeline](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline) to annotate the person of interest. 

Depending on your rendering backend, you may need to run with `MUJOCO_GL=egl` for the overlay video.

# Jupyter Notebook
A jupyter notebook with steps to run the pipeline can be found [here](https://github.com/IntelligentSensingAndRehabilitation/MonocularBiomechanics/blob/main/monocular-demo.ipynb).

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

