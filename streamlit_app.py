import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# Function to find closest color name
def find_closest_color(rgb, color_df):
    r, g, b = rgb
    differences = []
    for index, row in color_df.iterrows():
        try:
            # Parse R;G;B Dec format
            color_rgb = np.array([int(x) for x in row['R;G;B Dec'].split(';')])
            diff = np.sqrt(sum((rgb - color_rgb) ** 2))
            differences.append((diff, row['Color Name'], color_rgb))
        except (ValueError, KeyError):
            continue  # Skip rows with invalid RGB data
    if not differences:
        return "Unknown Color", []
    differences.sort(key=lambda x: x[0])
    closest = differences[0]
    return closest[1], differences[:3]  # Return color name and top 3 matches for debugging

# Main app
def main():
    st.title("Color Detection Application")
    st.write("Upload an image and click on it to detect colors")

    # Initialize session state
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'img_array' not in st.session_state:
        st.session_state.img_array = None
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = ""

    # Load color dataset
    try:
        color_df = pd.read_csv('colours_rgb_shades.csv')
        if not all(col in color_df.columns for col in ['Color Name', 'R;G;B Dec']):
            st.error("Invalid color dataset format. Required columns: 'Color Name', 'R;G;B Dec'")
            return
        # Clean dataset by removing rows with missing or invalid RGB values
        color_df = color_df[color_df['R;G;B Dec'].str.match(r'^\d+;\d+;\d+$', na=False)]
        if color_df.empty:
            st.error("No valid RGB data found in the dataset")
            return
    except FileNotFoundError:
        st.error("colours_rgb_shades.csv file not found")
        return
    except Exception as e:
        st.error(f"Error loading colours_rgb_shades.csv: {str(e)}")
        return

    # Image upload
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # Read and store image
            image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB format
            st.session_state.image = image
            img_array = np.array(image)
            st.session_state.img_array = img_array
            
            # Debug: Log image shape and sample pixel
            debug_info = f"Image shape: {img_array.shape}\n"
            debug_info += f"Sample pixel at (0,0): {img_array[0,0]}\n"
            
            # Display image with click functionality using streamlit-image-coordinates
            st.write("Click on the image to detect the color:")
            click_coords = streamlit_image_coordinates(image, key="image_click")
            
            # Process click if exists
            if click_coords:
                x, y = click_coords["x"], click_coords["y"]
                if 0 <= x < img_array.shape[1] and 0 <= y < img_array.shape[0]:
                    rgb = img_array[y, x]
                    debug_info += f"Click coordinates: ({x}, {y})\n"
                    debug_info += f"Extracted RGB at ({x}, {y}): {rgb}\n"
                    
                    # Find closest color
                    color_name, top_matches = find_closest_color(rgb, color_df)
                    debug_info += f"Closest color: {color_name}\n"
                    debug_info += "Top 3 matches:\n"
                    for diff, name, rgb_val in top_matches:
                        debug_info += f"- {name}: RGB {tuple(rgb_val)}, Distance: {diff:.2f}\n"
                    st.session_state.debug_info = debug_info
                    
                    # Display results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("**Detected Color Information**")
                        st.write(f"Color Name: {color_name}")
                        st.write(f"RGB Values: {tuple(rgb)}")
                        st.write("**Debug Information**")
                        st.write(debug_info)
                    
                    with col2:
                        # Display color box
                        color_box = np.zeros((100, 100, 3), dtype=np.uint8)
                        color_box[:] = rgb
                        color_box_rgb = cv2.cvtColor(color_box, cv2.COLOR_RGB2BGR)
                        st.image(color_box_rgb, caption="Selected Color", use_column_width=True)
                else:
                    st.error("Invalid click coordinates")
            
            else:
                st.session_state.debug_info = debug_info
                st.write("**Debug Information**")
                st.write(debug_info)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
