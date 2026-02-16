import os
import sys
import gc

# 1. WINDOWS DLL FIXES (Must be at the very top)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
anaconda_bin = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.exists(anaconda_bin) and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(anaconda_bin)

import streamlit as st
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# 2. TORCH IMPORT WITH GRACEFUL ERROR HANDLING
try:
    import torch
except OSError as e:
    st.error(f"DLL Loading Error: {e}. Please ensure Microsoft Visual C++ Redistributable is installed.")
    st.stop()

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import rbd
from lightglue import viz2d

# -------------------------
# Page Config & Styles
# -------------------------
st.set_page_config(page_title="Feature Matcher Pro", layout="wide")
st.title("🔍 Feature Extractor & Matcher Benchmark")
st.markdown("Comparing **SuperPoint**, **DISK**, and **SIFT** with accurate inference timing.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"Running on: **{device.type.upper()}**")

# -------------------------
# Session State Initialization
# -------------------------
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'models_warmed_up' not in st.session_state:
    st.session_state.models_warmed_up = set()

# -------------------------
# Helper Functions
# -------------------------
def clear_memory():
    """Forces garbage collection and clears Torch cache to prevent memory crashes."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@st.cache_resource
def load_models(name, _device, kpts):
    """Loads and caches models. Uses slightly lower depth for stability in Streamlit."""
    clear_memory()
    if name == "SuperPoint":
        extractor = SuperPoint(max_num_keypoints=kpts).eval().to(_device)
        matcher = LightGlue(features="superpoint", depth=7, prune=0.5).eval().to(_device)
    elif name == "DISK":
        extractor = DISK(max_num_keypoints=kpts).eval().to(_device)
        matcher = LightGlue(features="disk", depth=7, prune=0.5).eval().to(_device)
    else:
        return None, None
    return extractor, matcher

def warmup_model(extractor, matcher, _device, model_name):
    """Runs a small warmup inference to load CUDA kernels for accurate timing."""
    if model_name in st.session_state.models_warmed_up:
        return
    
    if _device.type == 'cuda' and extractor is not None:
        try:
            dummy = torch.zeros((1, 3, 64, 64)).to(_device)
            with torch.inference_mode():
                feats = extractor.extract(dummy)
                _ = matcher({"image0": feats, "image1": feats})
            torch.cuda.synchronize()
            st.session_state.models_warmed_up.add(model_name)
        except Exception:
            pass  # Warmup is optional, don't crash if it fails

def preprocess_image(img_file):
    """Resizes image to 800px max to ensure memory stability on Windows/Laptops."""
    img = Image.open(img_file).convert("RGB")
    w, h = img.size
    max_size = 800 
    scale = max_size / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img

def is_out_of_memory_error(e):
    """Checks if an exception is related to out-of-memory issues."""
    error_str = str(e).lower()
    return any(keyword in error_str for keyword in ['out of memory', 'oom', 'cuda error', 'allocation'])

def draw_matches_custom(img1, img2, kpts1, kpts2, color='#00FF00', line_thickness=1, keypoint_radius=3):
    """Custom match visualization that concatenates images and draws lines with keypoints."""
    # Convert hex to RGB
    hex_color = color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    color_rgb = (r, g, b)
    
    # Convert PIL to numpy if needed
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    
    # Get dimensions
    h1, w1 = img1_np.shape[:2]
    h2, w2 = img2_np.shape[:2]
    
    # Create combined image
    h_max = max(h1, h2)
    combined = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1_np
    combined[:h2, w1:w1+w2] = img2_np
    
    # Draw matches (lines first, then keypoints on top)
    for pt1, pt2 in zip(kpts1, kpts2):
        # Points in combined image coordinates
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]) + w1, int(pt2[1])
        
        # Draw line
        cv2.line(combined, (x1, y1), (x2, y2), color_rgb, line_thickness, cv2.LINE_AA)
    
    # Draw keypoints as circles on top of lines
    for pt1, pt2 in zip(kpts1, kpts2):
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]) + w1, int(pt2[1])
        
        # Draw filled circles for keypoints
        cv2.circle(combined, (x1, y1), keypoint_radius, color_rgb, -1, cv2.LINE_AA)
        cv2.circle(combined, (x2, y2), keypoint_radius, color_rgb, -1, cv2.LINE_AA)
    
    return combined

# -------------------------
# Sidebar Controls
# -------------------------
extractor_name = st.sidebar.selectbox("Choose Extractor", ["SuperPoint", "DISK", "SIFT"])
max_kpts = st.sidebar.slider("Max Keypoints", 512, 2048, 1024, step=512)
match_color = st.sidebar.color_picker("Match Line Color", "#00FF00")  # Lime green default
line_thickness = 1  # Fixed thickness - thin lines
show_all_matches = True  # Always show all matches
st.sidebar.divider()
st.sidebar.caption("💡 **Tip**: First run loads models (10-20s). Subsequent runs measure pure inference speed.")

# -------------------------
# Main Logic
# -------------------------
col1, col2 = st.columns(2)
with col1:
    img_file1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
with col2:
    img_file2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])

if img_file1 and img_file2:
    image1 = preprocess_image(img_file1)
    image2 = preprocess_image(img_file2)
    
    st.subheader("Input Images")
    c1, c2 = st.columns(2)
    c1.image(image1, caption=f"Image 1 ({image1.size[0]}×{image1.size[1]})", use_container_width=True)
    c2.image(image2, caption=f"Image 2 ({image2.size[0]}×{image2.size[1]})", use_container_width=True)

    if st.button("🚀 Run Matching", type="primary"):
        progress_bar = st.progress(0, text="Initializing...")
        
        try:
            # Step A: Load/Check Cache (Not Timed)
            progress_bar.progress(20, text="Loading models...")
            extractor, matcher = load_models(extractor_name, device, max_kpts)

            if extractor_name in ["SuperPoint", "DISK"]:
                # Warmup for accurate timing
                progress_bar.progress(30, text="Warming up GPU...")
                warmup_model(extractor, matcher, device, extractor_name)
                
                # Step B: Prepare Tensors (Not Timed)
                progress_bar.progress(40, text="Preparing images...")
                img1_t = torch.from_numpy(np.array(image1)).permute(2, 0, 1).float() / 255.0
                img2_t = torch.from_numpy(np.array(image2)).permute(2, 0, 1).float() / 255.0
                img1_t = img1_t.unsqueeze(0).to(device)
                img2_t = img2_t.unsqueeze(0).to(device)

                # Step C: ACTUAL INFERENCE (Timed)
                progress_bar.progress(60, text="Running inference...")
                if device.type == 'cuda': 
                    torch.cuda.synchronize()
                inference_start = time.time()

                with torch.inference_mode():
                    feats1 = extractor.extract(img1_t)
                    feats2 = extractor.extract(img2_t)
                    matches01 = matcher({"image0": feats1, "image1": feats2})

                if device.type == 'cuda': 
                    torch.cuda.synchronize()
                inference_time = time.time() - inference_start

                # Step D: Results Cleanup
                progress_bar.progress(80, text="Processing results...")
                feats1, feats2, matches01 = rbd(feats1), rbd(feats2), rbd(matches01)
                kpts1 = feats1["keypoints"][matches01["matches"][:, 0]].cpu().numpy()
                kpts2 = feats2["keypoints"][matches01["matches"][:, 1]].cpu().numpy()
                num_matches = len(kpts1)
                num_kpts1, num_kpts2 = len(feats1["keypoints"]), len(feats2["keypoints"])

            else:
                # SIFT Logic (Timed)
                progress_bar.progress(40, text="Running SIFT...")
                inference_start = time.time()
                sift = cv2.SIFT_create(nfeatures=max_kpts)
                gray1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)
                kp1, des1 = sift.detectAndCompute(gray1, None)
                kp2, des2 = sift.detectAndCompute(gray2, None)
                
                if des1 is not None and des2 is not None:
                    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
                    good_matches = [m for m, n in matches if len(matches) > 0 and m.distance < 0.75 * n.distance]
                else:
                    good_matches = []
                
                inference_time = time.time() - inference_start
                num_matches, num_kpts1, num_kpts2 = len(good_matches), len(kp1) if kp1 else 0, len(kp2) if kp2 else 0
                
                # Extract keypoint coordinates for SIFT
                if num_matches > 0:
                    kpts1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
                    kpts2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

            progress_bar.progress(100, text="Complete!")
            time.sleep(0.3)
            progress_bar.empty()

            # -------------------------
            # Metrics Display
            # -------------------------
            st.subheader("📊 Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pure Inference (s)", f"{inference_time:.3f}")
            m2.metric("Keypoints Img 1", num_kpts1)
            m3.metric("Keypoints Img 2", num_kpts2)
            m4.metric("Matches Found", num_matches)

            # -------------------------
            # Visualization
            # -------------------------
            st.subheader("🧩 Matching Visualization")
            
            if num_matches == 0:
                st.warning("⚠️ No matches found between the images. Try different images or adjust parameters.")
            else:
                # Use custom visualization for consistent rendering
                num_to_show = num_matches if (extractor_name == "SIFT" and show_all_matches) or extractor_name != "SIFT" else min(100, num_matches)
                
                # Create visualization
                matched_img = draw_matches_custom(
                    image1, image2, 
                    kpts1[:num_to_show], kpts2[:num_to_show], 
                    color=match_color,
                    line_thickness=line_thickness
                )
                
                # Display
                st.image(matched_img, caption=f"Showing {num_to_show} of {num_matches} matches", use_container_width=True)
                
                # Show match quality
                if num_matches > 0:
                    match_ratio = (num_matches / min(num_kpts1, num_kpts2)) * 100 if min(num_kpts1, num_kpts2) > 0 else 0
                    st.info(f"✅ Match ratio: {match_ratio:.1f}% of keypoints matched")

            # Store result in session state
            st.session_state.last_result = {
                'extractor': extractor_name,
                'time': inference_time,
                'matches': num_matches,
                'kpts': (num_kpts1, num_kpts2)
            }

        except RuntimeError as e:
            if is_out_of_memory_error(e):
                st.error("💥 **Out of Memory Error**")
                st.warning("The system ran out of RAM/VRAM. Try these solutions:")
                st.markdown("""
                - Reduce 'Max Keypoints' slider value
                - Close other applications
                - Use smaller images
                - Switch to CPU if using GPU
                """)
            else:
                st.error(f"⚠️ **Runtime Error**: {e}")
            clear_memory()
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"❌ **Unexpected Error**: {e}")
            st.info("Please check your inputs and try again.")
            clear_memory()
            progress_bar.empty()

else:
    st.info("⬆️ Please upload two images to begin benchmarking.")
    
    # Show example usage
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Upload Images**: Choose two images you want to match
        2. **Select Method**: Pick SuperPoint, DISK, or SIFT from the sidebar
        3. **Adjust Settings**: 
           - Max keypoints (higher = more features but slower)
           - Match line color (choose your preferred color)
        4. **Run**: Click the "Run Matching" button
        
        ### Understanding Results
        
        - **Pure Inference Time**: Actual algorithm speed (excludes loading)
        - **Keypoints**: Feature points detected in each image
        - **Matches**: Successfully matched points between images
        - **Match Ratio**: Percentage of keypoints that found a match
        
        ### Tips for Best Results
        
        - Images should have overlapping content
        - Good lighting and contrast help feature detection
        - SuperPoint: Best for general scenes, fast
        - DISK: Best for challenging viewpoints
        - SIFT: Classic method, reliable baseline
        """)

# Footer
st.sidebar.divider()
st.sidebar.caption("Built with LightGlue & Streamlit")
if st.session_state.last_result:
    st.sidebar.success(f"Last run: {st.session_state.last_result['matches']} matches in {st.session_state.last_result['time']:.3f}s")
