from PIL import Image
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
from io import BytesIO
import base64
import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from scipy.stats import rv_discrete
from skimage import color, io
from sklearn.cluster import KMeans, MiniBatchKMeans
import time
from functools import lru_cache
import urllib.parse
import io
import cv2





def color_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY])


def parse_contents(contents):
    print(f"Upload-image contents: {contents}")  # Add this line
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))  
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


# Function to load an image from a file
def load_default_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            image = Image.open(image_file)
            image = image.convert('RGB')  # Ensure it's in RGB format
            return image
    except IOError as error:
        print(f"Failed to load image from {image_path}: {error}")
        return None


# Convert image to data URL for web display
def image_to_data_url(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

# Test the function
def test_image_to_data_url():
    # Load an image file as a test
    test_image_path = "/Users/richard/Desktop/pythonprograms/SLIDER/assets/rainbow24bit.png"
    test_image = Image.open(test_image_path)

    # Generate data URL
    data_url = image_to_data_url(test_image)
    print(data_url)  # You can copy and paste this URL into a web browser to see if it displays the image correctly

test_image_to_data_url()



def calculate_unique_colors(image):
    # Convert the image to an array
    image_array = np.array(image)

    # Flatten the array and convert it to a list of tuples
    colors = list(map(tuple, image_array.reshape(-1, 3)))

    # Use a set to remove duplicate colors
    unique_colors = set(colors)

    # Return the number of unique colors
    return len(unique_colors)


def setup_image_and_marks():
    if default_image:
        image_str = image_to_data_url(default_image)

        # Calculate the number of unique colors in the image
        unique_colors = calculate_unique_colors(default_image)

        # Define marks based on unique colors
        marks = {
            1: {'label': 'Min', 'style': {'color': 'red', 'fontSize': '16px'}},
            unique_colors // 2: {'label': 'Mid', 'style': {'color': 'yellow', 'fontSize': '16px'}},
            unique_colors: {'label': str(unique_colors) + ' Max', 'style': {'color': 'green', 'fontSize': '16px'}}
        }

        return image_str, marks
    else:
        print("Image not loaded for marking.")
        return None, None


# Define the function before it's called
def generate_marks(unique_colors, max_colors):
    # Assuming max_colors and original_unique_colors are calculated before this function is called
    return {
        1: {'label': 'DownSample', 'style': {'color': 'red', 'fontSize': '16px', 'transform': 'translateX(-10%)'}},
        unique_colors: {'label': 'Original', 'style': {'color': 'blue', 'fontSize': '16px', 'transform': 'translateX(-50%)'}},
        max_colors: {'label': 'UpSample', 'style': {'color': 'green', 'fontSize': '16px', 'transform': 'translateX(-80%)'}}
    }


# Main script
default_image_path = "/Users/richard/Desktop/pythonprograms/SLIDER/assets/rainbow24bit.png"
default_image = load_default_image(default_image_path)
if default_image:
    unique_colors = calculate_unique_colors(default_image)
    max_colors = 10 * unique_colors  # Set max_colors to be 10x the number of unique colors
    marks = generate_marks(unique_colors, max_colors)
    print("Unique colors:", unique_colors)
    print("Marks for the slider:", marks)
else:
    print("Failed to load the image.")





def add_color(image, color):
    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Check if the image has an alpha channel
    if image_np.shape[-1] == 4:
        # Add an alpha value to the color
        color = np.append(color, 255)  # 255 is the maximum value for an 8-bit alpha channel

    # Create a new array with the same shape as image_np and fill it with the new color
    color_array = np.full(image_np.shape, color)

    # Add the new color to the image
    new_image_np = image_np + color_array

    # Convert the NumPy array back to an image
    new_image = Image.fromarray(new_image_np.astype('uint8'))

    return new_image



def is_too_similar(color, used_colors, threshold=30):
    return np.any(np.linalg.norm(np.array(color) - np.array(used_colors), axis=1) < threshold)



def quantize_image(image, n_colors):
    # Reshape the image to be a list of pixel values
    pixels = image.reshape(-1, 3)

    # Perform K-means clustering to find the most dominant colors
    kmeans = MiniBatchKMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    # Replace each pixel value with its nearest centroid
    new_pixels = kmeans.cluster_centers_[kmeans.labels_]
    new_image = new_pixels.reshape(image.shape)

    # Convert the image back to 8-bit representation
    new_image = np.clip(new_image, 0, 255).astype('uint8')

    # Calculate the unique colors after quantization
    unique_colors = len(np.unique(new_pixels, axis=0))
    print(f"Total unique colors now: {unique_colors}")

    return new_image






def pick_and_roll(image, offset, total_colors_requested):
    print("Starting pick_and_roll function...")
    num_pixels_recolored = 0  # Initialize the counter
    total_tries = 0  # Initialize the total number of tries

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    image_np = np.array(image)[:, :, :3]
    height, width, _ = image_np.shape
    pixels = image_np.reshape(-1, 3)
    
    total_colors_requested = max(1, int(total_colors_requested))
    unique_colors = len(set(map(tuple, pixels)))

    # Print initial color count
    print(f"Initial unique colors: {unique_colors}")

    num_new_colors = total_colors_requested - unique_colors

    # Optimize color counting to avoid multiple list operations
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    color_counts = {tuple(color): count for color, count in zip(unique, counts)}
    color_set = set(color_counts.keys())

    if num_new_colors > 0:
        new_colors = []  # Store all new colors
        for _ in range(num_new_colors):
            pixel_index = np.random.choice(len(pixels))
            old_color = pixels[pixel_index, :].copy()  # Store the old color

            # Only replace colors that have a count greater than 1
            if color_counts[tuple(old_color)] <= 1:
                continue

            # Determine the starting distance based on the number of unique colors
            starting_distance = max(1, unique_colors // 100)  # Adjust this formula as needed

            for i in range(256):  # Try up to 256 times to generate a new color
                total_tries += 1  # Increment the total number of tries

                # Generate a new color from the neighbors of the old color
                new_color = old_color + np.random.choice([-2, -1, 0, 1, 2], size=3)                
                new_color = np.clip(new_color, 0, 255)  # Make sure the color values are within the valid range

                # Update color_counts
                if tuple(new_color) not in color_counts:
                    color_counts[tuple(new_color)] = 0
                color_counts[tuple(new_color)] += 1
                color_counts[tuple(old_color)] -= 1

                # Ensure the comparison returns a scalar
                if np.linalg.norm(new_color - old_color) > 100:  # Check if the new color is too similar to the old color
                    continue

                pixels[pixel_index] = new_color
                num_pixels_recolored += 1  # Increment the counter
                new_colors.append(tuple(new_color))
                color_set.add(tuple(new_color))  # Update the set of colors
                break

    # Reshape pixels to the original image shape
    new_image_np = pixels.reshape(height, width, 3).astype(np.uint8)
    new_image = Image.fromarray(new_image_np)

    # Print the number of new colors added and the total colors
    print(f"New colors added: {num_pixels_recolored}")
    print(f"Total unique colors now: {len(color_set)}")

    # Print the average number of tries
    if num_new_colors > 0:
        print(f"Average number of tries: {total_tries / num_new_colors}")

    # Return the new image directly
    return new_image, num_pixels_recolored





# Now call setup_image_and_marks to initialize your image and marks
default_image_str, default_marks = setup_image_and_marks()
# default_image = load_default_image()
default_unique_colors = calculate_unique_colors(default_image)

# Define the CSS styles
css_styles = """
"""

# Create an empty figure
fig = go.Figure()


maximum_colors = 1000  # Arbitrary large number if you want a constant upper limit

# Assuming unique_colors has been calculated earlier
initial_value = unique_colors  # Set initial value of the slider to the number of unique colors




# Add the image to the layout
app.layout = html.Div([
    html.Img(id='my-image', src='/assets/placeholder.png'),  # Add a placeholder image
    dcc.Loading(
        id="loading",
        type="cube",
        children=[
            html.Link(
                rel='stylesheet',
                href='data:text/css;charset=UTF-8,' + urllib.parse.quote(css_styles),
            ),
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drag and Drop or ', html.A('Select Image')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px', 'borderColor': 'black',
                    'backgroundColor': '#2f2f2f', 'color': '#ffffff'
                },
                multiple=False
            ),
            html.Div(
                html.Img(id='image-display'),
                style={'display': 'flex', 'justifyContent': 'center'}
            ),
                html.Div(id='image-container', style={'display': 'flex', 'justifyContent': 'center'}),
            dcc.Slider(
                id='color-slider',
                min=1,
                max=maximum_colors,
                value=initial_value,
                step=1,
                marks=generate_marks(unique_colors, maximum_colors),
                tooltip={'always_visible': False, 'placement': 'top'}
            ),
            dcc.Store(id='roi-store'),  # For storing ROI data
            dcc.Store(id='unique-colors-store'),
            dcc.Store(id='offset-store'),  # For storing offset data
            dcc.Dropdown(
                id='version-dropdown',
                options=[
                    {'label': 'ColorBender V1', 'value': 'v1'},
                    {'label': 'ColorBender V2', 'value': 'v2'},
                    {'label': 'ColorBender V3', 'value': 'v3'},
                    {'label': 'ColorBender V4', 'value': 'v4'}
                ],
                value='v1',
                style={'backgroundColor': '#2f2f2f', 'color': '#ffffff', 'borderColor': '#2f2f2f'}
            ),
            html.Div([
                dcc.Graph(id='original-3d-bubble-plot', style={'width': '50%', 'height': '700px', 'display': 'inline-block', 'backgroundColor': '#2f2f2f'}),
                dcc.Graph(id='modified-3d-bubble-plot', style={'width': '50%', 'height': '700px', 'display': 'inline-block', 'backgroundColor': '#2f2f2f'})
            ], style={'width': '100%', 'display': 'flex', 'backgroundColor': '#2f2f2f'})
        ]
    )
])



def image_to_plot(image):
    # Convert the image to an array
    image_array = np.array(image)
    # Create a Plotly Heatmap
    return go.Figure(data=go.Heatmap(z=image_array, colorscale='gray'))



# Add a callback to handle the ROI selection
@app.callback(
    Output('roi-store', 'data'),
    Input('image-graph', 'relayoutData'),
)
def update_roi(relayoutData):
    # Extract the ROI from the relayoutData and store it in dcc.Store
    roi = extract_roi_from_relayoutData(relayoutData)
    return roi

def extract_roi_from_relayoutData(relayoutData):
    # If relayoutData is None, return a default ROI
    if relayoutData is None:
        return {'x': 0, 'y': 0, 'width': 0, 'height': 0}

    # Extract region of interest (roi) from relayoutData
    roi = {
        'x': relayoutData.get('xaxis.range[0]', 0),
        'y': relayoutData.get('yaxis.range[0]', 0),
        'width': relayoutData.get('xaxis.range[1]', 0) - relayoutData.get('xaxis.range[0]', 0),
        'height': relayoutData.get('yaxis.range[1]', 0) - relayoutData.get('yaxis.range[0]', 0),
    }
    return roi




def extract_roi_from_image(image, roi):
    # If roi is None or the width or height of the roi is 0, return the original image
    if roi is None or roi['width'] == 0 or roi['height'] == 0:
        return image
    
    # Convert the image to a numpy array
    image_np = np.array(image)

    # Get the image dimensions
    image_height, image_width = image_np.shape[:2]

    # Print the image dimensions and the ROI
    print(f"Image dimensions: {image_width}x{image_height}")
    print(f"ROI: {roi}")

    # Assume roi is a dictionary with keys 'x', 'y', 'width', 'height'
    x = int(roi['x'])
    y = int(roi['y'])
    width = int(roi['width'])
    height = int(roi['height'])

    # Ensure the ROI does not extend beyond the image boundaries
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    width = max(0, min(width, image_width - x))
    height = max(0, min(height, image_height - y))

    # Print the adjusted ROI
    print(f"Adjusted ROI: {{'x': {x}, 'y': {y}, 'width': {width}, 'height': {height}}}")

    # Extract the ROI from the image
    roi_image_np = image_np[y:y+height, x:x+width]

    # Convert the ROI back to a PIL Image and return it
    roi_image = Image.fromarray(roi_image_np)
    return roi_image




def count_unique_colors(image):
    # Reshape the image to be a 1D array of pixels
    pixels = image.reshape(-1, image.shape[-1])
    
    # Use numpy.unique to find unique rows (colors) in the array
    unique_colors = np.unique(pixels, axis=0)
    
    # The number of unique colors is the number of unique rows
    num_unique_colors = unique_colors.shape[0]
    
    return num_unique_colors



@app.callback(
    [Output('image-container', 'children'),
     Output('color-slider', 'max'),
     Output('color-slider', 'value'),
     Output('color-slider', 'marks'),
     Output('offset-store', 'data'),
     Output('unique-colors-store', 'data'),
     Output('image-display', 'src')],  # Add this line
    [Input('upload-image', 'contents'),
     Input('upload-image', 'filename'),
     Input('unique-colors-store', 'data')],
    [State('roi-store', 'data')]
)
def update_output(contents, filename, unique_colors_data, roi_data):
    if unique_colors_data is None:
        raise dash.exceptions.PreventUpdate
    
    print("update_output function triggered")
    # Initialize original_unique_colors
    original_unique_colors = None
    print("Starting update_output function...")
    
    # Get the ID of the component that triggered the callback
    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    print(f"Trigger ID: {trigger_id}")

    # Load the image from the uploaded contents or use the default image
    image = parse_contents(contents) if contents else load_default_image(default_image_path)

    # Convert the processed image to a data URL
    image_str = image_to_data_url(image)

    # Create an html.Img component with the data URL
    image_component = html.Img(src=image_str, style={'maxWidth': '100%', 'height': 'auto'})

    # Calculate the number of unique colors in the image
    unique_colors = calculate_unique_colors(image)
    print(f"Unique colors in the image: {unique_colors}")

    # Set the maximum value for the color slider
    max_colors = max(3, unique_colors * 10)  # Set max_colors to be 10x the number of unique colors
    print(f"Max colors: {max_colors}")

    # Generate the marks for the slider
    marks = generate_marks(unique_colors, max_colors)

    if trigger_id == 'upload-image':
        slider_value = unique_colors
        original_unique_colors = unique_colors  # Store the original unique colors
        print("Reset slider value to: {slider_value}")

    # Default value for offset
    offset = 0

    # Store the original number of unique colors before the downscaling operation
    original_unique_colors = unique_colors


    print(f"Slider value: {slider_value}, Unique colors: {unique_colors}")
    if slider_value and slider_value < unique_colors:
        start_time = time.time()  # Start the timer
        image = Image.fromarray(quantize_image(np.array(image), slider_value))
        unique_colors = calculate_unique_colors(image)
        end_time = time.time()  # Stop the timer
        print(f"calculate_unique_colors took {end_time - start_time} seconds")
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Downsampling took {elapsed_time} seconds.")
        print(f"Slider value: {slider_value}")
        downsample_label = f'Downsampling: {slider_value} colors'  # Update the downsampling label
    else:
        downsample_label = 'Downsampling: N/A'

    # If the slider value is more than the number of unique colors, blend the image with an offset image
    if slider_value and slider_value > unique_colors:
        offset = (slider_value - unique_colors) / unique_colors
        start_time = time.time()  # Start the timer
        try:
            # Upsample the image by adding new colors, and update the image with the result
            image, num_pixels_recolored = pick_and_roll(image, offset, slider_value)
            print(f"{slider_value} new colors have been requested.")
            print(f"{num_pixels_recolored} pixels have been recolored.")
            upsample_label = f'Upsampling: {slider_value} colors'  # Update the upsampling label
        except Exception as e:
            print(f"Error in pick_and_roll: {e}")
            upsample_label = 'Upsampling: Error'
        finally:
            end_time = time.time()  # Stop the timer
            elapsed_time = end_time - start_time  # Calculate the elapsed time
            print(f"Upsampling took {elapsed_time} seconds.")
    else:
        upsample_label = 'Upsampling: N/A'

    # Calculate the number of unique colors in the processed image
    unique_colors_in_processed_image = calculate_unique_colors(image)
    print(f"Number of total colors in processed image: {unique_colors_in_processed_image}")

    # Convert the image to a Plotly Heatmap
    figure = image_to_plot(image)

    # If the upload-image component triggered the callback, reset the slider value to the number of unique colors
    if trigger_id == 'upload-image':
        slider_value = unique_colors
        original_unique_colors = unique_colors  # Store the original unique colors
        print(f"Reset slider value to: {slider_value}")

    # If original_unique_colors is None, set it to unique_colors
    if original_unique_colors is None:
        original_unique_colors = unique_colors


    # Define the marks for the color slider
    marks = generate_marks(unique_colors)

    # Add a mark for the current slider value
    if slider_value < original_unique_colors:
        marks[slider_value] = {'label': str(slider_value), 'style': {'color': 'red', 'fontSize': '16px', 'transform': 'translateX(-50%)'}}
    elif slider_value == original_unique_colors:
        marks[slider_value] = {'label': str(slider_value), 'style': {'color': 'white', 'fontSize': '16px', 'transform': 'translateX(-50%)'}}
    else:  # slider_value > original_unique_colors
        marks[slider_value] = {'label': str(slider_value), 'style': {'color': 'green', 'fontSize': '16px', 'transform': 'translateX(-50%)'}}


    # Set the new children for the image-container, downsample-label, and upsample-label components
    image_container_children = html.Img(src=image_str, style={'maxWidth': '100%', 'height': 'auto'})

    # Return the outputs for the callback
    return image_container_children, maximum_colors, unique_colors_data, marks, offset_data, unique_colors_data, image_str  # Add image_str at the end


def create_3d_bubble_plot(image_np):
    # Ensure the image data is in the correct format (numpy array)
    if isinstance(image_np, Image.Image):
        image_np = np.array(image_np)  # Convert PIL Image to np.ndarray if necessary

    elif not isinstance(image_np, np.ndarray):
        raise ValueError("Unsupported image format. Expected np.ndarray or PIL Image.")

    # Ensure the image is in RGB format
    if len(image_np.shape) == 2:  # If the image is grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # If the image is RGBA
        pass  # Do nothing, keep the alpha channel
    elif image_np.shape[2] != 3:  # If the image is not RGB
        raise ValueError("The image must be an RGB or RGBA image.")

    # Check if the image has an alpha channel
    if image_np.shape[-1] == 4:
        # Extract color values from the image
        color = image_np[..., :3].reshape(-1, 3)
        # Add an alpha value to the color
        color = np.append(color, 255)  # 255 is the maximum value for an 8-bit alpha channel


    # When reshaping the image array and counting unique colors, take into account the number of channels in the image
    current_unique_colors = len(np.unique(image_np.reshape(-1, image_np.shape[-1]), axis=0))
    # Flatten the image to a list of RGB values
    print(image_np.shape)
    colors = image_np.reshape(-1, image_np.shape[-1])
    unique_colors, counts = np.unique(colors, axis=0, return_counts=True)

    # Calculate size of the markers based on color counts
    sizes = counts


    # Create a 3D scatter plot
    scatter = go.Scatter3d(
        x=unique_colors[:, 0],
        y=unique_colors[:, 1],
        z=unique_colors[:, 2],
        customdata=list(100 * sizes / image_np.size),
        mode='markers',
        marker=dict(
            size=sizes,
            sizemode='area',
            sizeref=2.*max(sizes)/(80.**2),
            sizemin=10,
            color=unique_colors / 255,  # normalize the color values
            opacity=0.8,
        ),
        hovertemplate='<b>R</b>: %{x}' +
                      '<br><b>G</b>: %{y}' +
                      '<br><b>B</b>: %{z}' +
                      '<br><b>Size</b>: %{marker.size}' +
                      '<br><b>Percent</b>: %{customdata:.4f}%'
    )

    # Define the layout with a dark theme
    layout = go.Layout(
        autosize=True,
        hovermode="closest",
        hoverdistance=1,
        uirevision=True,
        clickmode='event+select',
        paper_bgcolor="rgb(50, 50, 50)",  # Set dark background for the area around the plot
        font=dict(color="darkgray"),
        scene=dict(
            xaxis=dict(
                title="Red", 
                showspikes=False,
                backgroundcolor="black",
                gridcolor="lightgray",
                titlefont=dict(
                    color="red"
                ),
                tickfont=dict(
                    color="red"
                ),
            ),
            yaxis=dict(
                title="Green", 
                showspikes=False,
                backgroundcolor="black",
                gridcolor="lightgray",
                titlefont=dict(
                    color="green"
                ),
                tickfont=dict(
                    color="green"
                ),
            ),
            zaxis=dict(
                title="Blue", 
                showspikes=False,
                backgroundcolor="black",
                gridcolor="lightgray",
                titlefont=dict(
                    color="blue"
                ),
                tickfont=dict(
                    color="blue"
                ),
            ),
            bgcolor="rgb(50, 50, 50)"  # Set dark background for the plot
        ),
        modebar=dict(
            orientation='h',
            bgcolor='#31343a'
        ),
    )
    # Create the figure
    figure = go.Figure(data=[scatter], layout=layout)

    return figure




def prepare_image_for_plot(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    return image




@app.callback(
    [Output('original-3d-bubble-plot', 'figure'),
     Output('modified-3d-bubble-plot', 'figure')],
    [Input('upload-image', 'contents'),
     Input('color-slider', 'value')],
    [State('roi-store', 'data')]
)
def update_bubble_plots(contents, slider_value, roi):
    # Check if slider_value is a dictionary
    if isinstance(slider_value, dict):
        # Extract the current value of the slider from the dictionary
        slider_value = slider_value[str(default_unique_colors)]['label']

    # Ensure slider_value is an integer
    try:
        print(f"Before conversion, slider_value is: {slider_value}")
        slider_value = int(slider_value)
    except ValueError as e:
        print(f"Error converting slider_value to int: {e}")
        return None, None

    if contents:
        try:
            image_str = contents.split(',')[1]  # Extract the base64 string from the contents
            image_bytes = base64.b64decode(image_str)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            print(f"Failed to load image from contents: {e}")
            return None, None
    else:
        image = load_default_image(default_image_path)
        if image is None:
            print("Failed to load default image.")
            return None, None

    image_np = np.array(image)

    # Extract the ROI from the image
    try:
        roi_image = extract_roi_from_image(image, roi)
    except Exception as e:
        print(f"Failed to extract ROI: {e}")
        return None, None

    # Use the ROI for creating the 3D bubble plot
    original_figure = create_3d_bubble_plot(roi_image)

    # Determine the number of unique colors in the current image
    current_unique_colors = len(np.unique(image_np.reshape(-1, 3), axis=0))

    if slider_value < current_unique_colors:
        # Downsampling: reduce the number of colors
        processed_image_np = quantize_image(image_np, slider_value)
    elif slider_value > current_unique_colors:
        # Upsampling: increase the number of colors
        processed_image, _ = pick_and_roll(Image.fromarray(image_np), 0, slider_value)
        processed_image_np = np.array(processed_image)
    else:
        processed_image_np = image_np

    modified_figure = create_3d_bubble_plot(processed_image_np)

    return original_figure, modified_figure







if __name__ == '__main__':
    app.run_server(debug=True)
