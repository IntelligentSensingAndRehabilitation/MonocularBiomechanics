from monocular_demos.utils import jax_memory_limit, tensorflow_memory_limit
tensorflow_memory_limit()
jax_memory_limit()
import os
from typing import List

os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import gradio as gr
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from monocular_demos.biomechanics_mjx.forward_kinematics import ForwardKinematics
from monocular_demos.biomechanics_mjx.visualize import render_trajectory, render_overlay
from monocular_demos.biomechanics_mjx.monocular_trajectory import (
    fit_model,
    get_model,
)
from monocular_demos.utils import load_metrabs, joint_names, video_reader
from monocular_demos.dataset import MonocularDataset,get_samsung_calibration

fk = ForwardKinematics(
    xml_path="monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml",
)

jax.config.update("jax_compilation_cache_dir", "./.jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_enable_x64", True)


def save_metrabs_data(accumulated, video_name):
    fname = video_name.split("/")[-1].split(".")[0]
    boxes, pose3d, pose2d, confs = [], [], [], []
    for i, (box, p3d, p2d) in enumerate(
        zip(accumulated["boxes"], accumulated["poses3d"], accumulated["poses2d"])
    ):
        # TODO: write logic for better box tracking
        if len(box) == 0:
            boxes.append(np.zeros((5)))
            pose3d.append(np.zeros((87, 3)))
            pose2d.append(np.zeros((87, 2)))
            confs.append(np.zeros((87)))
            print("no boxes")
            continue
        boxes.append(box[0].numpy())
        pose3d.append(p3d[0].numpy())
        pose2d.append(p2d[0].numpy())
        confs.append(np.ones((87)))

        with open(f"{fname}_keypoints.npz", "wb") as f:
            np.savez(f, keypoints3d=pose3d, keypoints2d=pose2d, boxes=boxes, confs=confs)

def find_matching_video(fname):
    """Find a video file that matches the given filename with any video extension"""
    video_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV']
    
    for ext in video_extensions:
        video_path = f"{fname}{ext}"
        if os.path.exists(video_path):
            return video_path
    
    return None

def render_mjx(selected_file, progress=gr.Progress()):
    """Load saved data and create visualizations"""
    if not selected_file or selected_file == "No fitted models found":
        return "Please select a fitted model file first.", None, None
    
    fname = selected_file.replace('_fitted_model.npz', '')
    
    # Try to load keypoints data
    biomech_file = selected_file
    
    result_text = ""
    video_filename = f"{fname}_mjx.mp4"
    
    if os.path.exists(biomech_file):
        with open(biomech_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            result_text += f"Loaded biomechanics data: {biomech_file}\n"
            qpos = data['qpos']
    progress(0, desc="Rendering Video (progress will not display linearly)...")
    render_trajectory(
        qpos,
        filename = video_filename,
        xml_path="monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque_vis.xml",
        height=800,
        width=800,
    )
    progress(1.0, desc="Visualization complete!")
    result_text += f"Rendered visualization: {video_filename}\n"

    return result_text, video_filename

def render_overlay_video(selected_file, max_frames, progress=gr.Progress()):
    """Load saved data and create visualizations"""
    if not selected_file or selected_file == "No fitted models found":
        return "Please select a fitted model file first.", None
    
    fname = selected_file.replace('_fitted_model.npz', '') # TODO: fix this
    
    # Try to load keypoints data
    biomech_file = selected_file
    
    result_text = ""
    
    if os.path.exists(biomech_file):
        with open(biomech_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            result_text += f"Loaded biomechanics data: {biomech_file}\n"
            fit_timestamps = data['timestamps']
            qpos = data['qpos']
            body_scale = data['scale']

            # Find matching video file with any extension
            video_path = find_matching_video(fname)
            if video_path is None:
                return f"No matching video file found for {fname}", None
            
            result_text += f"Found video file: {video_path}\n"
            
            # Get video dimensions dynamically
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return f"Could not open video file: {video_path}", None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
    else:
        return "No biomechanics data found for the selected file.", None
    
    progress(0, desc="Rendering Overlay...")
    
    overlay = render_overlay(
        fit_timestamps,
        qpos,
        body_scale,
        video_path,
        width,
        height,
        progress,
        max_frames=max_frames if max_frames > 0 else None,
    )
    
    progress(1.0, desc="Overlay rendering complete!")
    
    return result_text, overlay


def get_framerate(video_path):
    """
    Get the framerate of a video file.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def load_metrabs_data(video_name):
    fname = video_name.split("/")[-1].split(".")[0]
    try:
        with open(f"{fname}_keypoints.npz", "rb") as f:
            data = np.load(f, allow_pickle=True)
            boxes = data["boxes"]
            keypoints2d = data["keypoints2d"]
            keypoints3d = data["keypoints3d"]
            confs = data["confs"]

        return boxes, keypoints2d, keypoints3d, confs
    except FileNotFoundError:
        print("No saved data found for this video.")
        return None, None, None, None


def process_videos_with_metrabs(
    video_files: List[str],
    progress=gr.Progress(),
) -> str:
    """
    Process the uploaded videos. Replace this with your actual processing logic.
    """
    if not video_files:
        return "No videos uploaded."

    progress(0, desc="Loading model (takes 3 minutes)...")
    model = load_metrabs()
    progress(0.1, desc="Model loaded successfully.")
    skeleton = "bml_movi_87"

    video_count = 0
    for video_idx, video_path in enumerate(video_files):
        if video_path is not None:
            vid, n_frames = video_reader(video_path)
            accumulated = None
            for frame_idx, frame_batch in enumerate(vid):
                progress(
                    frame_idx * 8 / n_frames, desc=f"Processing video {video_idx+1}"
                )

                pred = model.detect_poses_batched(frame_batch, skeleton=skeleton)

                if accumulated is None:
                    accumulated = pred

                else:
                    # concatenate the ragged tensor along the batch for each element in the dictionary
                    for key in accumulated.keys():
                        accumulated[key] = tf.concat(
                            [accumulated[key], pred[key]], axis=0
                        )
            save_metrabs_data(accumulated, video_path)
            video_count += 1

    return f"Successfully processed {video_count} videos with Metrabs."


def process_videos_with_biomechanics(
    video_files: List[str], progress=gr.Progress()
) -> str:
    """
    Process the uploaded videos with biomechanics fitting. Replace this with your actual processing logic.
    """
    import equinox as eqx
    import jax
    jax.clear_caches()
    eqx.clear_caches()

    max_iters = 10000

    def step_callback(step, model, dataset, metrics_dict, **kwargs):
        if step % 500 == 0:
            progress(step / max_iters, desc=f"Fitting model: Step {step}/{max_iters}")

    if not video_files:
        return "No videos uploaded."

    timestamps_list = []
    keypoints2d_list = []
    keypoints3d_list = []
    confs_list = []
    for i, video_path in enumerate(video_files):
        if video_path is not None:
            boxes, keypoints2d, keypoints3d, confs = load_metrabs_data(video_path)
            if boxes is None:
                print(f"Video {video_path}: No Metrabs data found.")
                continue

            fps = get_framerate(video_path)
            timestamps = np.arange(0, len(keypoints2d)) / fps
            timestamps_list.append(timestamps)
            keypoints2d_list.append(keypoints2d[jnp.newaxis])  # Add camera dimension
            keypoints3d_list.append(keypoints3d[jnp.newaxis])  # Add camera dimension
            confs_list.append(
                jnp.ones_like(keypoints2d[..., 0])[jnp.newaxis]
            )  # fake confidences

    dataset = MonocularDataset(
        timestamps=timestamps_list,
        keypoints_2d=keypoints2d_list,
        keypoints_3d=keypoints3d_list,
        keypoint_confidence=confs_list,
        camera_params=get_samsung_calibration(),
        phone_attitude=None,
    )

    progress(0, desc="Building biomechanics model...")
    model = get_model(
        dataset, xml_path="monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml", joint_names=joint_names
    )  # might need to change the site names
    model, metrics = fit_model(
        model,
        dataset,
        lr_init_value=1e-3,
        max_iters=max_iters,
        step_callback=step_callback,
    )
    progress(1.0, desc="Biomechanics model fit successfully.")

    for i, video_path in enumerate(video_files):
        progress(
            i / len(video_files),
            desc=f"Saving biomechanics for video {i+1}/{len(video_files)}.",
        )
        timestamps = dataset.get_all_timestamps(i)

        (state, _, _), (qpos, qvel, _), rnc = model(
            timestamps,
            trajectory_selection=i,
            steps=0,
            skip_action=True,
            fast_inference=True,
            check_constraints=False,
        )

        # save zip archive
        fname = video_path.split("/")[-1].split(".")[0]
        with open(f"{fname}_fitted_model.npz", "wb") as f:
            np.savez(
                f,
                timestamps=np.array(timestamps),
                qpos=np.array(qpos),
                qvel=np.array(qvel),
                rnc=np.array(rnc),
                sites=np.array(state.site_xpos),
                joints=np.array(state.xpos),
                scale=np.array(model.body_scale)
            )

    return f"Successfully processed {len(dataset)} videos with biomechanics fitting."


def get_available_fitted_models():
    """Get list of available fitted model files"""
    fitted_files = [f for f in os.listdir('.') if f.endswith('_fitted_model.npz')]
    return fitted_files if fitted_files else ["No fitted models found"]

def load_and_visualize_data(selected_file, selected_joints=None):
    """Load saved data and create visualizations"""
    if not selected_file or selected_file == "No fitted models found":
        return "Please select a fitted model file first.", None, None
    
    selected_joint_inds = [fk.joint_names.index(joint) for joint in selected_joints] if selected_joints else []
    
    fname = selected_file.replace('_fitted_model.npz', '')
    
    # Try to load keypoints data
    keypoints_file = f"{fname}_keypoints.npz"
    biomech_file = selected_file
    
    result_text = ""
    plot1 = None
    
    if os.path.exists(keypoints_file):
        with open(keypoints_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            result_text += f"Loaded keypoints data: {keypoints_file}\n"
            result_text += f"- Frames: {len(data['keypoints3d'])}\n"
            result_text += f"- 3D keypoints shape: {data['keypoints3d'].shape}\n"
            result_text += f"- 2D keypoints shape: {data['keypoints2d'].shape}\n\n"
    else:
        result_text += f"No keypoints data found for {fname}\n\n"
    
    if os.path.exists(biomech_file):
        with open(biomech_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            result_text += f"Loaded biomechanics data: {biomech_file}\n"
            result_text += f"- Timesteps: {len(data['qpos'])}\n"
            result_text += f"- Joint positions shape: {data['qpos'].shape}\n"
            result_text += f"- Joint velocities shape: {data['qvel'].shape}\n"

            # Create Plot 1: Joint angles over time
            qpos = data['qpos']
            time_steps = np.arange(len(qpos))
            
            fig1 = go.Figure()
            # Plot selected joints
            if selected_joints:
                for joint_idx in selected_joint_inds:
                    if joint_idx < qpos.shape[1]:
                        fig1.add_trace(go.Scatter(
                            x=time_steps, 
                            y=qpos[:, joint_idx], 
                            mode='lines',
                            name=f'Joint {fk.joint_names[joint_idx]}'
                        ))
            else:
                # Default to first 6 joints if none selected
                for i in range(min(6, qpos.shape[1])):
                    fig1.add_trace(go.Scatter(
                        x=time_steps, 
                        y=qpos[:, i], 
                        mode='lines',
                        name=f'Joint {i+1}'
                    ))
            fig1.update_layout(
                title="Joint Angles Over Time",
                xaxis_title="Time Steps",
                yaxis_title="Angle (radians)"
            )
            plot1 = fig1
    else:
        result_text += f"No biomechanics data found for {fname}\n"
    
    return result_text, plot1

def get_joint_options(selected_file):
    """Get available joint options for the selected model"""
    if not selected_file or selected_file == "No fitted models found":
        return gr.Dropdown(choices=[], value=[])
    
    biomech_file = selected_file
    
    if os.path.exists(biomech_file):
        with open(biomech_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            num_joints = data['qpos'].shape[1]
            joint_choices = [(f"Joint {i+1}", i) for i in range(num_joints)]
            default_selection = list(range(min(6, num_joints)))  # Default to first 6
            
            return gr.Dropdown(
                choices=joint_choices,
                value=default_selection,
                multiselect=True
            )
    
    return gr.Dropdown(choices=[], value=[])

def clear_videos():
    """Clear the video upload component"""
    return None


# Create the Gradio interface
with gr.Blocks(title="Open Portable Biomechanics Lab") as demo:
    gr.Markdown("# Open Portable Biomechanics Lab")

    with gr.Tab("Processing"):
        gr.Markdown(
            "Upload multiple videos for processing. Supported formats: MP4, AVI, MOV, MKV"
        )
        with gr.Row():
            with gr.Column():
                video_input = gr.File(
                    label="Upload Videos",
                    file_count="multiple",
                    file_types=["video"],
                    height=200,
                )

                with gr.Row():
                    metrabs_btn = gr.Button("1. Keypoint Detection", variant="primary")
                    mjx_btn = gr.Button("2. Biomechanical Fitting", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(
                    label="Processing Results", lines=10, max_lines=20, interactive=False
                )

        # Event handlers
        metrabs_btn.click(
            fn=process_videos_with_metrabs,
            inputs=[video_input],
            outputs=[output_text],
        )

        mjx_btn.click(
            fn=process_videos_with_biomechanics, inputs=[video_input], outputs=[output_text]
        )

        # Also process when files are uploaded
        video_input.change(
            fn=lambda files: f"Uploaded {len(files) if files else 0} videos. Click 'Process Videos' to continue.",
            inputs=[video_input],
            outputs=[output_text],
        )

    with gr.Tab("Kinematic Plots"):
        gr.Markdown("Visualize and analyze processed data")
        with gr.Row():
            with gr.Column():

                # Dropdown with all fitted model files
                fitted_model_dropdown = gr.Dropdown(
                    choices=get_available_fitted_models(),
                    label="Select Fitted Model",
                    value=None,
                    interactive=True
                )

                joint_selection_dropdown = gr.Dropdown(
                    choices=fk.joint_names,
                    label="Select Joints to Plot",
                    multiselect=True,
                    value=[],
                    interactive=True
                )
                
                refresh_btn = gr.Button("Refresh File List", variant="secondary")

                load_data_btn = gr.Button("Load Data", variant="primary")
                
            with gr.Column():
                viz_info = gr.Textbox(
                    label="Data Information",
                    lines=10,
                    interactive=False
                )
                
                # Placeholder for plots - you can expand these
                viz_plot1 = gr.Plot(label="Visualization")
        
        # Event handlers for visualization tab
        refresh_btn.click(
            fn=lambda: gr.Dropdown(choices=get_available_fitted_models()),
            outputs=[fitted_model_dropdown]
        )
        
        load_data_btn.click(
            fn=load_and_visualize_data,
            inputs=[fitted_model_dropdown, joint_selection_dropdown],
            outputs=[viz_info, viz_plot1]
        )
    
    with gr.Tab("Visualization"):
        with gr.Row():
            with gr.Column():

                # Dropdown with all fitted model files
                fitted_model_dropdown = gr.Dropdown(
                    choices=get_available_fitted_models(),
                    label="Select Fitted Model",
                    value=None,
                    interactive=True
                )

            max_frames_input = gr.Number(
                label="Max Frames",
                value=10,
                minimum=0,
                maximum=1000,
                step=1,
                info="Maximum number of frames to render (0 for all frames)"
            )

            with gr.Column():
                viz_info = gr.Textbox(
                    label="Data Information",
                    lines=10,
                    interactive=False
                )
        
        visualize_btn = gr.Button("Render Visualization", variant="primary")

        # create video viewer
        video_viewer = gr.Video(label="Visualization Video", autoplay=True, height=400)

        refresh_btn.click(
            fn=lambda: gr.Dropdown(choices=get_available_fitted_models()),
            outputs=[fitted_model_dropdown]
        )
        
        visualize_btn.click(
            fn=render_overlay_video,
            inputs=[fitted_model_dropdown, max_frames_input],
            outputs=[viz_info, video_viewer]
        )


demo.launch()
