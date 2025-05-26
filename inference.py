import os
import re
from dual_encoder import DualEncoderModel
import paired_datasets.paired_image_dataset as paired
import torch
from functools import partial
import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd


def extract_lat_lon_from_path(path):
    # works for names like .../1234_45.1_8.9_0.jpg
    basename = os.path.basename(path)
    match = re.search(r'_(-?\d+\.\d+)_(-?\d+\.\d+)_', basename)
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        return lat, lon
    else:
        return None, None

        
def inference_dual_encoder(
        batch_size: int = 16,
        device: str = "cuda",
        output_csv: str = "comb_predictions.csv",
):
    
    checkpoint = 'microsoft/swin-base-patch4-window7-224'
    # 5) Load dual encoder model
    model = DualEncoderModel(checkpoint=checkpoint)  # Replace with your actual class name
    model.load_state_dict(torch.load('dual-encoder/dual-encoder.pt'))
    model.to(device).eval()

    
    dataset = paired.StreetSatDataset()
    dataset.download_and_prepare()
    dataset_splits = dataset.as_dataset()

    
    # Split del dataset (80% train, 10% valid e 10% test)
    tf = paired.create_transforms(mode="both",checkpoint=checkpoint)
    inference_split = dataset_splits["inference"].with_transform(tf)
    street_paths = [p['street'] for p in inference_split]

    # 4) DataLoader with collate_fn
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
            street = batch['street'].to(device)
            sat = batch['sat'].to(device)
            out = model(street=street, sat=sat)
            logits = out.logits if hasattr(out, "logits") else out
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            # extract lat/lon from batch images
            batch_street_paths = batch.get('street_paths', None) or batch.get('paths', None)
            if batch_street_paths is None:  # fallback to global
                batch_street_paths = street_paths[len(all_coords):len(all_coords)+len(preds)]
            for path in batch_street_paths:
                lat, lon = extract_lat_lon_from_path(path)
                all_coords.append((lat, lon))

    # 6) Aggregate predictions by (lat, lon)
    coord2preds = defaultdict(list)
    for coord, pred in zip(all_coords, all_preds):
        coord2preds[coord].append(pred)

    # Majority vote (or mean) for each point
    final_rows = []
    for (lat, lon), preds in coord2preds.items():
        final_class = round(np.mean(preds))
        final_rows.append({
            "lat": lat,
            "lon": lon,
            "class": final_class + 1,  # If your classes start at 1, as in your code
        })

    df = pd.DataFrame(final_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} predictions → {output_csv}")

if __name__ == '__main__':
    inference_dual_encoder()