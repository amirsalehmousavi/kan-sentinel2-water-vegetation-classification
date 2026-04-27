import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
import seaborn as sns
from osgeo import gdal
import rasterio
import matplotlib.font_manager as fm
from rasterio.transform import from_origin
import csv
import torch
import os
from .KANLayer import KANLinear

def tiff(file_path):
    ''' Reads a tiff file and stores it into a python array '''
    dataset = gdal.Open(file_path)
    band_data = []
    num_bands = dataset.RasterCount
    for i in range(1, num_bands + 1):
        band = dataset.GetRasterBand(i)
        band_data.append(band.ReadAsArray())
    return np.dstack(band_data)


def visualize_image(image_array, title):
    ''' Visualize the output of the tiff function '''
    plt.figure(figsize=(5, 5))
    plt.imshow(image_array)
    plt.title(title)
    plt.axis('off')
    plt.show()


def tiff_data(file_path):
    ''' Reads the .tiff file, and prints a summary on each of its bands

    Parameters:
    - file_path: path to the .tiff file

    Returns:
    - Prints a summary of the .tiff image '''

    dataset = gdal.Open(file_path)
    if dataset is None:
        print("Failed to open the file.")
        return None

    band_data = []
    num_bands = dataset.RasterCount
    for i in range(1, num_bands + 1):
        band = dataset.GetRasterBand(i)
        if band is None:
            print(f"Failed to read band {i}")
            return None

        data = band.ReadAsArray()
        band_data.append(data)

        ## Print data type and some statistics
        print(f"Band {i}:")
        print(f" Data Type: {gdal.GetDataTypeName(band.DataType)}")
        print(f" Min: {data.min()}, Max: {data.max()}")
        print(f" Mean: {data.mean()}, StdDev: {data.std()}")
        print(f" Sample pixel value: {data[0, 0]}")

    return np.dstack(band_data)


def existing_nodata_values(file_path):
    ''' Reads an image and stores all of the nodata values in a python dictionary

    Parameters:
    - file_path: path to the .tiff file

    Returns:
    - nodata_values: python dictionary containing all the no-data values '''

    ## Open the dataset
    dataset = gdal.Open(file_path)

    ## Dictionary to store unique nodata values for each band
    nodata_values = {}

    ## Iterate over each band and check for the nodata value
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        nodata_value = band.GetNoDataValue()

        ## Store nodata values in a set for each band
        if i not in nodata_values:
            nodata_values[i] = set()
        nodata_values[i].add(nodata_value)

    ## Print unique nodata values for each band
    for band_index, values in nodata_values.items():
        # Remove None from set if it exists
        values.discard(None)
        print(f'Band {band_index} nodata values: {values}')

    return nodata_values


def read_tiff(file_path):
    dataset = gdal.Open(file_path)
    band_data = []
    for i in range(1, dataset.RasterCount + 1):  # Read all bands
        band = dataset.GetRasterBand(i)
        band_data.append(band.ReadAsArray())
    return np.dstack(band_data)


def remove_nodata(image, nodata_values=[-99.0], remove_zeros=True):
    mask = np.ones(image.shape[:2], dtype=bool)
    for nodata_value in nodata_values:
        mask &= np.all(image != nodata_value, axis=2)

    if remove_zeros:
        mask &= np.any(image != 0, axis=2)

    valid_pixels = image[mask]
    return valid_pixels


def process_single_file(file_path, label):
    img = read_tiff(file_path)
    valid_data = remove_nodata(img)
    pixels = valid_data.reshape(-1, valid_data.shape[-1])
    labels = np.full((pixels.shape[0], 1), label, dtype=int)
    return pixels, labels



def plot_activations(model, input_range=(-3, 3), num_points=1000, save_dir='activations'):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device

    input_size = model.kan1.in_features
    prev_layer_notation = 0

    for layer_idx, layer in enumerate(model.children()):
        if isinstance(layer, KANLinear):
            current_input_size = input_size
            input_size = layer.out_features

            for node_idx in range(layer.out_features):
                fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

                x = torch.linspace(input_range[0], input_range[1], num_points).to(device)

                for dim in range(current_input_size):
                    input_tensor = torch.zeros(num_points, current_input_size).to(device)
                    input_tensor[:, dim] = x

                    with torch.no_grad():
                        base_output = layer.base_activation(input_tensor)
                        base_weight = layer.base_weight[node_idx:node_idx + 1, :]
                        base_activation = torch.mm(base_output, base_weight.t())

                        spline_bases = layer.b_splines(input_tensor)
                        spline_weight = layer.scaled_spline_weight[node_idx:node_idx + 1, :, :]
                        spline_activation = torch.mm(
                            spline_bases.view(num_points, -1),
                            spline_weight.view(1, -1).t()
                        )

                        total_activation = base_activation + spline_activation

                    ax.plot(x.cpu().numpy(), total_activation.cpu().numpy(),
                            label=f'$\\varphi_{{{prev_layer_notation},{dim+1},{node_idx+1}}}$')

                ax.set_title(f'Activation Functions for Layer {layer_idx + 1}, Node {node_idx + 1}',
                             fontsize=14, fontweight='bold')
                ax.set_xlabel('Input', fontsize=12)
                ax.set_ylabel('Activation', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.tick_params(axis='both', which='major', labelsize=10)

                # Move legend outside the plot
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'activation_layer{layer_idx + 1}_node{node_idx + 1}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close(fig)

            prev_layer_notation += 1

    print(f"Activation plots for all layers saved in '{save_dir}'")


def compare_masks(mlp_mask, kan_mask, original_image, cmap,name):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    rgb_image = original_image[:, :, [0, 1, 2]]
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))

    ax1.imshow(rgb_image)
    ax1.set_title(f'Original Image (false colors) - {name}')
    ax1.axis('off')

    ax2.imshow(mlp_mask, cmap=cmap)
    ax2.set_title(f'MLP Classification - {name}')
    ax2.axis('off')

    ax3.imshow(kan_mask, cmap=cmap)
    ax3.set_title(f'KAN Classification - {name}')
    ax3.axis('off')

    plt.savefig(f'Comparison_{name}.png', bbox_inches='tight')
    plt.show()
    plt.close()

def create_mask(predictions, original_shape):
        mask = predictions.reshape(original_shape[:2])
        return mask

def save_mask(mask, filename, cmap):
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap=cmap)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

# Function to save the mask as a georeferenced GeoTIFF
def save_mask_geotiff(mask, filename, original_tif_path):
    # Read georeferencing information from the original TIFF file
    with rasterio.open(original_tif_path) as src:
        transform = src.transform
        crs = src.crs
        dtype = src.dtypes[0]

    # Write the mask as a GeoTIFF with the same georeferencing information
    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

def classify_pixels(model, pixels, chunk_size=10000):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, pixels.shape[0], chunk_size):
            chunk = pixels[i:i+chunk_size]
            outputs = model(chunk)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.cpu().numpy())
    return np.concatenate(predictions)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

def print_metrics(model_name, y_true, y_pred, class_names, area_name='general'):
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Prepare CSV file
    csv_filename = f'{area_name}_{model_name}_metrics.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write and print model name and kappa
        print(f"\n{model_name} Results:")
        print(f"Kappa Coefficient: {kappa:.4f}")
        csvwriter.writerow([f"{model_name} Results"])
        csvwriter.writerow(["Kappa Coefficient", f"{kappa:.4f}"])

        # Write and print classification report
        print("\nClassification Report:")
        csvwriter.writerow([])
        csvwriter.writerow(["Classification Report"])
        for class_name in class_names:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1_score = report[class_name]['f1-score']
            support = report[class_name]['support']

            omission_error = 1 - recall
            commission_error = 1 - precision

            print(f"\n{class_name}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-score: {f1_score:.4f}")
            print(f"  Omission Error: {omission_error:.4f}")
            print(f"  Commission Error: {commission_error:.4f}")
            print(f"  Support: {support}")

            csvwriter.writerow([class_name])
            csvwriter.writerow(["Precision", f"{precision:.4f}"])
            csvwriter.writerow(["Recall", f"{recall:.4f}"])
            csvwriter.writerow(["F1-score", f"{f1_score:.4f}"])
            csvwriter.writerow(["Omission Error", f"{omission_error:.4f}"])
            csvwriter.writerow(["Commission Error", f"{commission_error:.4f}"])
            csvwriter.writerow(["Support", support])

        # Write and print confusion matrix
        print("\nConfusion Matrix:")
        csvwriter.writerow([])
        csvwriter.writerow(["Confusion Matrix"])
        for row in cm:
            print(" ".join(f"{val:5d}" for val in row))
            csvwriter.writerow(row)

    # Plotting confusion matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=False, cmap='GnBu', cbar=True)
    plt.ylabel('True', fontsize = 14)
    plt.xlabel('Predicted', fontsize = 14)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Annotate the confusion matrix with the values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = 'white' if value > np.max(cm) / 2 else 'black'
            plt.text(j + 0.5, i + 0.5, str(value), horizontalalignment='center', verticalalignment='center',
                     color=color, fontsize = 12)

    # Setting class names on the axes
    plt.xticks(ticks=np.arange(cm.shape[1]) + 0.5, labels=class_names, size = 12)
    plt.yticks(ticks=np.arange(cm.shape[0]) + 0.5, labels=class_names, size = 12)
    plt.savefig(f'{area_name}_{model_name}_confusion_matrix', pad_inches=0, dpi=600)

    plt.show()

    print(f"Metrics saved to {csv_filename}")
