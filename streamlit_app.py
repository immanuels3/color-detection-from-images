import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import os

# Function to find closest color name
def find_closest_color(rgb, color_df):
    r, g, b = rgb
    differences = []
    for index, row in color_df.iterrows():
        try:
            color_rgb = np.array([int(x) for x in row['R;G;B Dec'].split(';')])
            diff = np.sqrt(sum((rgb - color_rgb) ** 2))
            differences.append((diff, row['Color Name']))
        except (ValueError, KeyError):
            continue
    return min(differences)[1] if differences else "Unknown Color"

# Load color dataset
@st.cache_data
def load_color_dataset():
    try:
        color_df = pd.read_csv('colours_rgb_shades.csv')
        if not all(col in color_df.columns for col in ['Color Name', 'R;G;B Dec']):
            raise ValueError("Invalid color dataset format. Required columns: 'Color Name', 'R;G;B Dec'")
        color_df = color_df[color_df['R;G;B Dec'].str.match(r'^\d+;\d+;\d+$', na=False)]
        if color_df.empty:
            raise ValueError("No valid RGB data found in the dataset")
        return color_df
    except FileNotFoundError:
        st.error("colours_rgb_shades.csv file not found")
        return None
    except Exception as e:
        st.error(f"Error loading colours_rgb_shades.csv: {str(e)}")
        return None

# Process a single frame for color detection
def process_frame(frame, color_df):
    # Convert to HSV
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges
    red_lower1 = np.array([0, 120, 70], np.uint8)
    red_upper1 = np.array([10, 255, 255], np.uint8)
    red_lower2 = np.array([170, 120, 70], np.uint8)
    red_upper2 = np.array([180, 255, 255], np.uint8)
    green_lower = np.array([40, 40, 40], np.uint8)
    green_upper = np.array([80, 255, 255], np.uint8)
    blue_lower = np.array([100, 150, 0], np.uint8)
    blue_upper = np.array([140, 255, 255], np.uint8)

    # Create masks
    red_mask1 = cv2.inRange(hsvFrame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsvFrame, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Dilate masks
    kernel = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

    # List to store unique detected colors (color_name, rgb)
    detected_colors = set()

    # Function to process contours and label colors
    def process_contours(mask, box_color):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                avg_color = np.mean(roi, axis=(0, 1)).astype(int)
                b, g, r = avg_color
                rgb = np.array([r, g, b])
                color_name = find_closest_color(rgb, color_df)
                
                # Add to detected colors (using tuple to make RGB hashable)
                detected_colors.add((color_name, tuple(rgb)))
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, f"{color_name}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color)

    # Process each color
    process_contours(red_mask, (0, 0, 255))
    process_contours(green_mask, (0, 255, 0))
    process_contours(blue_mask, (255, 0, 0))

    # Add detected colors at the bottom of the image
    if detected_colors:
        # Sort detected colors by color name for consistency
        detected_colors = sorted(detected_colors, key=lambda x: x[0])
        color_text = " | ".join([f"{color_name} {rgb}" for color_name, rgb in detected_colors])
        
        # Calculate the position for the text
        h, w, _ = frame.shape
        text_size, _ = cv2.getTextSize(color_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_w, text_h = text_size
        
        # Draw a semi-transparent background rectangle for the text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - text_h - 20), (w, h), (0, 0, 0), -1)  # Black background
        alpha = 0.5  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw the text
        cv2.putText(frame, color_text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame

# Main app
def main():
    st.title("Multiple Color Detection Application")
    st.write("Upload an image or video to detect colors (red, green, blue regions).")

    # Load color dataset
    color_df = load_color_dataset()
    if color_df is None:
        return

    # File upload
    uploaded_file = st.file_uploader("Choose an image or video", type=['png', 'jpg', 'jpeg', 'mp4', 'avi'])

    if uploaded_file is not None:
        # Determine file type
        file_type = uploaded_file.type
        is_image = file_type.startswith('image')

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4' if not is_image else '.png') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            if is_image:
                # Process as image
                image = cv2.imread(tmp_file_path)
                if image is None:
                    st.error("Failed to load image")
                    return
                processed_frame = process_frame(image, color_df)
                st.image(processed_frame, channels="BGR", caption="Detected Colors")
            else:
                # Process as video
                cap = cv2.VideoCapture(tmp_file_path)
                if not cap.isOpened():
                    st.error("Failed to load video")
                    return

                stframe = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed_frame = process_frame(frame, color_df)
                    stframe.image(processed_frame, channels="BGR", caption="Detected Colors")
                cap.release()

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

if __name__ == "__main__":
    main()
