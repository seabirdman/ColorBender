from PIL import Image
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
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


def color_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY])

default_image_path = '/Users/richard/Desktop/pythonprograms/SLIDER/assets/rainbow24bit.png'


def load_default_image():
    try:
        with open(default_image_path, "rb") as image_file:
            return Image.open(BytesIO(image_file.read()))
    except Exception as e:
        print(f"Error loading default image: {e}")
        return None
    
def parse_contents(contents):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(BytesIO(decoded))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def image_to_data_url(image):
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

@lru_cache(maxsize=None)
def calculate_unique_colors(image_bytes, image_size):
    image = Image.frombytes('RGB', image_size, image_bytes)
    data = np.array(image.convert('RGB')).reshape(-1, 3)
    return len(np.unique(data, axis=0))

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

    if num_new_colors > 0:
        new_colors = []  # Store all new colors
        color_set = set(map(tuple, pixels))  # Create a set of colors
        for _ in range(num_new_colors):
            pixel_index = np.random.choice(len(pixels))
            old_color = pixels[pixel_index, :].copy()  # Store the old color

            if tuple(old_color) not in color_counts:
                print(f"Skipping iteration because old_color is not in color_counts: old_color = {old_color}")
                continue

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







def setup_image_and_marks():
    # Load the default image and calculate its unique colors
    image = load_default_image()
    unique_colors = calculate_unique_colors(image.tobytes(), image.size)
    image_str = image_to_data_url(image)

    # Set the slider marks based on the image properties
    marks = {
        1: {'label': 'Min', 'style': {'color': 'red', 'fontSize': '16px'}},
        unique_colors: {'label': str(unique_colors), 'style': {'color': 'white', 'fontSize': '16px'}},
        unique_colors * 2: {'label': 'Max', 'style': {'color': 'green', 'fontSize': '16px'}}
    }

    return image_str, marks

# Use the function to setup the image and marks
default_image_str, default_marks = setup_image_and_marks()
default_image = load_default_image()
default_unique_colors = calculate_unique_colors(default_image.tobytes(), default_image.size)

# logo_email_link = "mailto:podolsky@att.net?subject=ColorBender%20Question"

app.layout = dash.dcc.Loading(
    id="loading",
    type="cube",
    children=[
        html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drag and Drop or ', html.A('Select Image')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px', 'borderColor': 'black'
                },
                multiple=False
            ),
            html.Div(id='image-container', children=html.Img(src=default_image_str, style={'maxWidth': '100%', 'height': 'auto'}), style={'textAlign': 'center'}),
            dcc.Slider(
                id='color-slider',
                min=1,
                max=default_unique_colors * 2,
                value=default_unique_colors,
                step=1,
                marks=default_marks,
                tooltip={'always_visible': False, 'placement': 'top'}
            ),
            dcc.Store(id='offset-store'),



            html.Div([
                dcc.Graph(id='original-3d-bubble-plot', style={'width': '50%', 'height': '700px', 'display': 'inline-block', 'backgroundColor': '#2f2f2f'}),
                dcc.Graph(id='modified-3d-bubble-plot', style={'width': '50%', 'height': '700px', 'display': 'inline-block', 'backgroundColor': '#2f2f2f'})
            ], style={'width': '100%', 'display': 'flex', 'backgroundColor': '#2f2f2f'})


            
        ])
    ]
)


def count_unique_colors(image):
    # Reshape the image to be a 1D array of pixels
    pixels = image.reshape(-1, image.shape[-1])
    
    # Use numpy.unique to find unique rows (colors) in the array
    unique_colors = np.unique(pixels, axis=0)
    
    # The number of unique colors is the number of unique rows
    num_unique_colors = unique_colors.shape[0]
    
    return num_unique_colors



@app.callback(
    Output('color-slider', 'max'),
    Output('color-slider', 'value'),
    Output('color-slider', 'marks'),
    Output('offset-store', 'data'),
    Output('image-container', 'children'),
    Input('upload-image', 'contents'),
    Input('upload-image', 'filename'),
    Input('color-slider', 'value'),
    State('offset-store', 'data')
)
def update_output(contents, filename, slider_value, offset):
    print("update_output function triggered")
    # Initialize original_unique_colors
    original_unique_colors = None
    print("Starting update_output function...")
    
    # Get the ID of the component that triggered the callback
    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    print(f"Trigger ID: {trigger_id}")

    # Load the image from the uploaded contents or use the default image
    image = parse_contents(contents) if contents else load_default_image()

    # Calculate the number of unique colors in the image
    unique_colors = calculate_unique_colors(image.tobytes(), image.size)
    print(f"Unique colors: {unique_colors}")

    # Set the maximum value for the color slider
    max_colors = max(3, unique_colors * 5)
    print(f"Max colors: {max_colors}")

    # If the upload-image component triggered the callback, reset the slider value to the number of unique colors
    if trigger_id == 'upload-image':
        start_time = time.time()  # Start the timer
        slider_value = unique_colors
        end_time = time.time()  # Stop the timer
        print(f"Reset slider value to: {slider_value}")
        print(f"Resetting slider value took {end_time - start_time} seconds")
        
    # Default value for offset
    offset = 0

    # Store the original number of unique colors before the downscaling operation
    original_unique_colors = unique_colors

    print(f"Slider value: {slider_value}, Unique colors: {unique_colors}")
    if slider_value and slider_value < unique_colors:
        start_time = time.time()  # Start the timer
        image = Image.fromarray(quantize_image(np.array(image), slider_value))
        unique_colors = calculate_unique_colors(image.tobytes(), image.size)
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
    unique_colors_in_processed_image = calculate_unique_colors(image.tobytes(), image.size)
    print(f"Number of total colors in processed image: {unique_colors_in_processed_image}")

    # Convert the processed image to a data URL
    image_str = image_to_data_url(image)

    # If the upload-image component triggered the callback, reset the slider value to the number of unique colors
    if trigger_id == 'upload-image':
        slider_value = unique_colors
        original_unique_colors = unique_colors  # Store the original unique colors
        print(f"Reset slider value to: {slider_value}")

    # If original_unique_colors is None, set it to unique_colors
    if original_unique_colors is None:
        original_unique_colors = unique_colors


    # Define the marks for the color slider
    marks = {
        1: {'label': 'DownSample', 'style': {'color': 'red', 'fontSize': '16px', 'transform': 'translateX(-10%)'}},
        max_colors: {'label': 'UpSample', 'style': {'color': 'green', 'fontSize': '16px', 'transform': 'translateX(-80%)'}},
        original_unique_colors: {'label': 'Original', 'style': {'color': 'blue', 'fontSize': '16px', 'transform': 'translateX(-50%)'}}
    }

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
    return max_colors, slider_value, marks, offset, image_container_children











def create_3d_bubble_plot(image_np):
    # Ensure the image data is in the correct format (numpy array)
    if isinstance(image_np, Image.Image):
        image_np = np.array(image_np)  # Convert PIL Image to np.ndarray if necessary

    elif not isinstance(image_np, np.ndarray):
        raise ValueError("Unsupported image format. Expected np.ndarray or PIL Image.")

    # Ensure the image is in RGB format
    if image_np.shape[-1] != 3:
        raise ValueError("The image must be an RGB image with three channels.")

    # Flatten the image to a list of RGB values
    colors = image_np.reshape(-1, 3)
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
    [State('image-container', 'children')]  # Use state to get the default image if no upload
)
def update_bubble_plots(contents, slider_value, image_container):
    if contents:
        image = parse_contents(contents)
    else:
        image = load_default_image()

    image_np = np.array(image)

    # Original plot: Use original image data
    original_figure = create_3d_bubble_plot(image_np)

    # Determine the number of unique colors in the current image
    current_unique_colors = len(np.unique(image_np.reshape(-1, 3), axis=0))

    if slider_value < current_unique_colors:
        # Downsampling: reduce the number of colors using quantize_image
        processed_image_np = quantize_image(image_np, slider_value)
    elif slider_value > current_unique_colors:
        # Upsampling: increase the number of colors using pick_and_roll
        # Note: pick_and_roll expects an Image object and returns a tuple, so we adjust accordingly
        processed_image, _ = pick_and_roll(Image.fromarray(image_np), 0, slider_value)
        processed_image_np = np.array(processed_image)
    else:
        processed_image_np = image_np

    modified_figure = create_3d_bubble_plot(processed_image_np)

    return original_figure, modified_figure









if __name__ == '__main__':
    app.run_server(debug=True)
