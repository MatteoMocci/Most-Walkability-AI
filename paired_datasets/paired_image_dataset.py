import numpy as np
from datasets import DatasetInfo, GeneratorBasedBuilder, SplitGenerator, Split
from datasets.features import Features, Image, ClassLabel
from transformers import AutoFeatureExtractor, AutoImageProcessor
import torch

'''
Questo è un dataset custom per caricare le immagini streetview e satellitari.
'''


class StreetSatDataset(GeneratorBasedBuilder):
    '''
    Questo dataset ha tre modalità:
    1) Street, ogni istanza del dataset è un immagine streetview
    2) Sat, ogni istanza del dataset è un immagine satellitare
    3) Both, ogni istanza del dataset ha sia un immagine streetview e la corrispondente immagine satellitare
    '''
    def __init__(self, mode='both'):
        self.mode = mode
        super().__init__()
    
    def _info(self):
        features = {
            "label": ClassLabel(names=["0", "1", "2", "3", "4"])    # le etichette vanno da 0 a 4
        }

        # Le colonne street e sat vengono create solo nelle modalite che necessitano
        if self.mode in ["both", "street"]:
            features["street"] = Image()
        
        if self.mode in ["both", "sat"]:
            features["sat"] = Image()
        
        return DatasetInfo(
            features=Features(features)
        )
    
    '''
    Questa funzione genera gli split di training, validation e test. Questi split sono stati predefiniti e salvati
    in formato .npy. Questo per garantire che lo split sarà sempre uguale e che le immagini nello split di training del dataset
    in modalità "street" siano le stesse del dataset in modalità "sat". E' importante che, nella combinazione
    dell'output dei modelli, i due modelli abbiano avuto le stesse immagini di input se addestrati coi due dataset separati.
    '''
    def _split_generators(self, dl_manager):
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"file_path": "paired_datasets/train.npy"},
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={"file_path": "paired_datasets/valid.npy"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"file_path": "paired_datasets/test.npy"},
            ),
            SplitGenerator(
                name="inference",
                gen_kwargs={"file_path": "paired_datasets/inference.npy"},
            )
        ]
    

    def _generate_examples(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        for idx, pair in enumerate(data):
            image_1_path, image_2_path = pair
            label = image_1_path.split('/')[0] # La label è rappresentata dalla cartella in cui è contenuta l'immagine
            if label.isnumeric(): # Se la label è numerica, allora si tratta del dataset originale per lo split
            
            # nei path manca la cartella
                image_1_path = 'dataset-Sardegna180/' + image_1_path
                image_2_path = 'satellite/' + image_2_path
            
            else: #altrimenti è il set per l'inferenza, la label non serve, metti 0
                label = 0

            # A seconda della modalità viene inserita l'immagine corrispondente e la label
            example = {'label': label}
            if self.mode in ["both", "street"]:
                example["street"] = image_1_path
            if self.mode in ["both", "sat"]:
                example["sat"] = image_2_path
            yield idx, example 

'''
Questa funzione crea una funzione transform a seconda della modalità. Se la modalità è street o sat, verrà processata solo l'immagine
corrispondente, usando come chiave 'pixel_values'. Altrimenti, entrambe le immagini vengono processate.
'''
def create_transforms(checkpoint,mode="both"):
    def transforms(batch):
        if 'dino' in checkpoint:
            processor = AutoImageProcessor.from_pretrained(checkpoint, ignore_mismatched_sizes=True)
        else:
            processor = AutoFeatureExtractor.from_pretrained(checkpoint, ignore_mismatched_sizes=True)
        inputs = {}
        if mode == "street" and "street" in batch:
            batch["street"] = [x.convert('RGB') for x in batch["street"]]
            inputs['pixel_values'] = processor(batch["street"], return_tensors='pt')['pixel_values']
        elif mode == "sat" and "sat" in batch:
            batch["sat"] = [x.convert('RGB') for x in batch["sat"]]
            inputs['pixel_values'] = processor(batch["street"], return_tensors='pt')['pixel_values']
        elif mode == "both":
            inputs = {}
            if "street" in batch:
                batch["street"] = [x.convert('RGB') for x in batch["street"]]
                inputs["street"] = processor(batch["street"], return_tensors='pt')['pixel_values']
            if "sat" in batch:
                batch["sat"] = [x.convert('RGB') for x in batch["sat"]]
                inputs["sat"] = processor(batch["sat"], return_tensors='pt')['pixel_values']
        if "label" in batch:
            inputs["label"] = batch["label"]
        return inputs
    return transforms

'''
Questa funzione definisce come aggregare un batch di esempi, a seconda della modalità selezionata.
Da 'label' si passa a 'labels'.
'''
def collate_fn(batch, mode="both"):
    output = {}
    if mode in ["both", "street"] and "street" in batch[0]:
        output["street"] = torch.stack([x["street"] for x in batch])
    if mode in ["both", "sat"] and "sat" in batch[0]:
        output["sat"] = torch.stack([x["sat"] for x in batch])
    if 'pixel_values' in batch[0]:
        output['pixel_values'] = torch.stack([x['pixel_values'] for x in batch])
    output['labels'] = torch.tensor([x['label'] for x in batch])
    return output
    

