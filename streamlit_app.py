import gradio as gr
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io

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
    return closest[1], differences[:3]

# Function to process image click
def process_image_click(image_file, evt: gr.SelectData):
    if image_file is None:
        return None, "No image uploaded", None, "No debug info"
    
    try:
        # Read the image directly from the file to avoid Gradio's processing
        image = Image.open(image_file).convert("RGB")
        img_array = np.array(image)
        
        # Log the image shape and a sample pixel
        debug_info = f"Image shape: {img_array.shape}\n"
        debug_info += f"Sample pixel at (0,0): {img_array[0,0]}\n"
        
        # Get click coordinates
        x, y = evt.index[0], evt.index[1]
        debug_info += f"Click coordinates: ({x}, {y})\n"
        
        # Validate coordinates
        if not (0 <= x < img_array.shape[1] and 0 <= y < img_array.shape[0]):
            return None, "Invalid click coordinates", None, debug_info
        
        # Extract RGB value
        rgb = img_array[y, x]
        debug_info += f"Extracted RGB at ({x}, {y}): {rgb}\n"
        
        # Load color dataset
        try:
            color_df = pd.read_csv('colours_rgb_shades.csv')
            if not all(col in color_df.columns for col in ['Color Name', 'R;G;B Dec']):
                return None, "Invalid color dataset format. Required columns: 'Color Name', 'R;G;B Dec'", None, debug_info
            color_df = color_df[color_df['R;G;B Dec'].str.match(r'^\d+;\d+;\d+$', na=False)]
            if color_df.empty:
                return None, "No valid RGB data found in the dataset", None, debug_info
        except FileNotFoundError:
            return None, "colours_rgb_shades.csv file not found", None, debug_info
        except Exception as e:
            return None, f"Error loading colours_rgb_shades.csv: {str(e)}", None, debug_info
        
        # Find closest color
        color_name, top_matches = find_closest_color(rgb, color_df)
        debug_info += f"Closest color: {color_name}\n"
        debug_info += "Top 3 matches:\n"
        for diff, name, rgb_val in top_matches:
            debug_info += f"- {name}: RGB {tuple(rgb_val)}, Distance: {diff:.2f}\n"
        
        # Create color box
        color_box = np.zeros((100, 100, 3), dtype=np.uint8)
        color_box[:] = rgb
        color_box_rgb = cv2.cvtColor(color_box, cv2.COLOR_RGB2BGR)
        
        # Format output text
        output_text = f"**Detected Color Information**\n\nColor Name: {color_name}\nRGB Values: {tuple(rgb)}"
        
        return Image.fromarray(color_box_rgb), output_text, None, debug_info
    
    except Exception as e:
        return None, f"Error processing image: {str(e)}", None, "Error in processing"

# Main Gradio interface
def main():
    with gr.Blocks(title="Color Detection Application") as demo:
        gr.Markdown("# Color Detection Application")
        gr.Markdown("Upload an image and click on it to detect colors.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="Upload Image (PNG, JPG, JPEG)")
            with gr.Column():
                color_box_output = gr.Image(label="Selected Color")
                text_output = gr.Textbox(label="Color Information")
                error_output = gr.Textbox(label="Errors", visible=True)
                debug_output = gr.Textbox(label="Debug Information", visible=True)
        
        # Set up click event
        image_input.select(
            process_image_click,
            inputs=[image_input],
            outputs=[color_box_output, text_output, error_output, debug_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch(server_name="0.0.0.0", server_port=7860)
