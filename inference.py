# Custom modules import
from dual_encoder import DualEncoderModel
import paired_datasets.paired_image_dataset as paired

# Import of single items
from collections import defaultdict
from functools import partial
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree
from tqdm import tqdm

# Standard modules import
import os
import re
import pandas as pd
import numpy as np
import torch


'''
This is an utility function for extracting latitude and longitude from a name file. In the inference dataset, files are usually named like this
number_lat_lon_number.jpg. Extracting lat and lon will be useful to create an output shape file where each location is associated with the corresponding model prediction.
'''
def extract_lat_lon_from_path(path):
    basename = os.path.basename(path)

    # Find two (possibly negative) decimal numbers separated by underscores in 'basename'
    match = re.search(r'_(-?\d+\.\d+)_(-?\d+\.\d+)_', basename) 

    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        return lat, lon
    else:
        return None, None


'''
This is an utility function to obtain a list of all images contained in root_dir. Each element of the list is a tuple containing the lat, the lon and the full_path.
'''
def get_image_list(root_dir):
    image_list = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png'):
                full_path = os.path.join(root, f)
                lat, lon = extract_lat_lon_from_path(f)
                if lat is not None and lon is not None:
                    image_list.append((lat, lon, full_path))
    return image_list


'''
Starting from two folders, street_dir and sat_dir, this code builds a kdTree with the aim to pair each element from street_dir with the closest element of sat_dir. Each photo in street_dir and sat_dir has been taken in a specific location identified by (lat,lon) coordinates.
The result of this function is the saving of a .npy file of pairs. Each pair contains firstly the path of a streetview picture and then the path of the closest corresponding satellite picture. Other than the two folders, the other parameters represent the path for the
output npy file and a value of maximum offset tolerance for matching streetview and sat: if the distance is > than tolerance_m there will be no match.
'''
def combine_photos_kdtree(street_dir, sat_dir, out_npy='paired_datasets/inference.npy', tolerance_m=10):


    # Take the streetview pictures from street_dir and the satellite from sat_dir
    print(f"Parsing streetview images from {street_dir} ...")
    street_images = get_image_list(street_dir)
    print(f"Parsing satellite images from {sat_dir} ...")
    sat_images = get_image_list(sat_dir)

    # Ends the program if one of the two lists is empty
    if len(street_images) == 0 or len(sat_images) == 0:
        print("No images found in one of the directories.")
        return

    # Builds a KDtree from the satellite coordinates
    sat_coords = np.array([[lat, lon] for lat, lon, _ in sat_images])
    sat_paths = [f for _, _, f in sat_images]
    tree = cKDTree(sat_coords)

    # Query each street image on the KDTree, thus obtaining the closest satellite image and create a new pair.
    pairs = []
    for lat_s, lon_s, path_s in street_images:
        dist, idx = tree.query([lat_s, lon_s], k=1)

        dist_m = dist * 111_000         # 1 degree latitude ≈ 111_000 meters
        if dist_m <= tolerance_m:
            path_t = sat_paths[idx]
            pairs.append((path_s, path_t))

    # Save the paires in the output .npy file
    print(f"Paired {len(pairs)} streetview images with satellite images (within {tolerance_m} meters).")
    np.save(out_npy, pairs)
    print(f"Saved pairs to {out_npy}.")

'''
This function executes inference on a dataset using the dual encoder model. The dual encoder model consists of two Vision Transformer, one that is trained on streetview images and one that is trained on satellite imagery. Each data point of the inference area is represented through a
streetview and satellite picture. First, the model is loaded, then the .npy containing the file paths of the relevant images is loaded and a dataset is created. A dataloader creates batches of the inference pictures. Since each point has two pair of photos (one in the front, one in the back),
the predictions are averaged. Then, a csv file is created with the model predictions. The parameters of this method are the batch_size (number of instances to work at a time), device (CPU or gpu), and the csv file where to output predictions.
'''
def inference_dual_encoder(
        batch_size: int = 16,
        device: str = "cuda",
        output_csv: str = "comb_predictions.csv",
):
    
    checkpoint = 'microsoft/swin-base-patch4-window7-224' # This is the huggingface's checkpoint of the original pre-trained network

    # Load the state of the pre-trained model
    model = DualEncoderModel(checkpoint=checkpoint)  
    model.load_state_dict(torch.load('dual-encoder/dual-encoder.pt'))
    model.to(device).eval()

    # Fetch the dataset for inference
    dataset = paired.StreetSatDataset()
    dataset.download_and_prepare()
    dataset_splits = dataset.as_dataset()
 
    # Apply transform to all dataset pictures
    tf = paired.create_transforms(mode="both",checkpoint=checkpoint)
    inference_split = dataset_splits["inference"].with_transform(tf)
    street_paths = [p['street'] for p in inference_split]

    # Create dataloader for batching the inference pictures
    loader = DataLoader(
        inference_split,
        batch_size=batch_size,
        collate_fn=partial(paired.collate_fn, mode="both"),
        shuffle=False,
    )

    all_preds = []
    all_coords = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):

            # for each batch, extract the model's prediction
            street = batch['street'].to(device)
            sat = batch['sat'].to(device)
            out = model(street=street, sat=sat)
            
            # Extract logits if available and assemble predictions
            if isinstance(out, dict):
                logits = out["logits"]
            elif hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
        
            # extract lat/lon from batch images
            batch_street_paths = batch.get('street_paths', None) or batch.get('paths', None)
            if batch_street_paths is None:  # fallback to global
                batch_street_paths = street_paths[len(all_coords):len(all_coords)+len(preds)]
            for path in batch_street_paths:
                lat, lon = extract_lat_lon_from_path(path)
                all_coords.append((lat, lon))

    # Aggregate predictions by (lat, lon)
    coord2preds = defaultdict(list)
    for coord, pred in zip(all_coords, all_preds):
        coord2preds[coord].append(pred)

    # Save the data that will be stored in the prediction csv
    final_rows = []
    for (lat, lon), preds in coord2preds.items():
        final_class = round(np.mean(preds))
        final_rows.append({
            "lat": lat,
            "lon": lon,
            "class": final_class + 1,  # If your classes start at 1, as in your code
        })

    # Create a dataframe with the prediction data and convert it to csv
    df = pd.DataFrame(final_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} predictions → {output_csv}")

if __name__ == '__main__':
    inference_dual_encoder()