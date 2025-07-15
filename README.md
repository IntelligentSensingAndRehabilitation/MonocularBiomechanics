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
The way a person moves is a direct reflection of their neurological and musculoskeletal health, yet it remains one of the most underutilized vital signs in clinical practice. Although clinicians visually observe movement impairments, they lack accessible and validated methods to objectively measure movement in routine care. This gap prevents wider use of biomechanical measurements in practice, which could enable more sensitive outcome measures or earlier identification of impairment.

In this work, we present our Portable Biomechanics Laboratory (PBL), which includes a secure, cloud-enabled smartphone app for data collection and a novel algorithm for fitting biomechanical models to this data. We extensively validated PBL’s biomechanical measures using a large, clinically representative and heterogeneous dataset with synchronous ground truth. Next, we tested the usability and utility of our system in both a neurosurgery and sports medicine clinic.

We found joint angle errors within 3 degrees and pelvis translation errors within several centimeters across participants with neurological injury, lower-limb prosthesis users, pediatric inpatients and controls. In addition to being easy and quick to use, gait metrics computed from the PBL showed high reliability (ICCs > 0.9) and were sensitive to clinical differences. For example, in individuals undergoing decompression surgery for cervical myelopathy, the modified Japanese Orthopedic Association (mJOA) score is a common patient-reported outcome measure; we found that PBL gait metrics not only correlated with mJOA scores but also demonstrated greater responsiveness to surgical intervention than the patient-reported outcomes.

These findings support the use of handheld smartphone video as a scalable, low-burden, tool for capturing clinically meaningful biomechanical data, offering a promising path toward remote, accessible monitoring of mobility impairments in clinical populations. To promote further research and clinical translation, we open-source the first method for measuring whole-body kinematics from handheld smartphone video validated in clinical populations: [https://github.com/IntelligentSensingAndRehabilitation/MonocularBiomechanics](https://github.com/IntelligentSensingAndRehabilitation/MonocularBiomechanics)

<video src="docs/static/videos/jd_running.mp4" width="800" controls autoplay muted loop></video>

# Code
Clone and install
```
git clone git@github.com:IntelligentSensingAndRehabilitation/MonocularBiomechanics.git
cd MonocularBiomechanics/
pip install -e .
```
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
