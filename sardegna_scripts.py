import torch
import numpy as np
from datasets import load_dataset, DatasetDict
import evaluate 
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments, pipeline
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import osmnx as ox
import math

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", ignore_mismatched_sizes=True)

# FUNZIONI PRINCIPALI

'''
Questo script carica le immagini dalla cartella dataset-Sardegna180 e crea un dataset diviso in train (80%), validation (10%) e test (10%).
La label di ciascuna immagine è direttamente determinata dal nome della cartella a cui essa appartiene. La funzione restituisce i tre split dopo aver estratto le feature di ogni immagine
'''
def load_data():
    dataset = load_dataset("imagefolder", data_dir= "C:/Users/mocci/Desktop/MOST/Sardegna180/dataset-Sardegna180", split='train').train_test_split(test_size = 0.2)
    train_testvalid = dataset['train'].train_test_split(test_size=0.2)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})
    train = dataset['train'].with_transform(transform)
    valid = dataset['valid'].with_transform(transform)
    test = dataset['test'].with_transform(transform)
    return train, valid, test

'''
Questa funzione setta i parametri del Trainer e lo inizializza.
@param train    il training set
@param valid    il validation set
@param n        il numero di epoche per cui effettuare il training
@return il trainer creato
'''
def create_trainer(train,valid,n,to_train=True,lr=2e-4,optim="adamw_torch",batch_size=16,checkpoint="google/vit-base-patch16-224"):
    if to_train:
        model = AutoModelForImageClassification.from_pretrained(checkpoint, num_labels=5, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForImageClassification.from_pretrained("AEnigmista/Sardegna-ViT", num_labels=5, ignore_mismatched_sizes=True)

    training_args = TrainingArguments(
    output_dir="./sardegna-vit",
    per_device_train_batch_size=batch_size,
    evaluation_strategy="steps",
    num_train_epochs=n,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=100,
    learning_rate=lr,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
    use_cpu=False,
    optim=optim
    )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=processor,
    )
    return trainer

'''
Questa funzione addestra il modello
@param trainer il modello da addestrare
'''
def train_model(trainer):
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    #trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    return trainer

'''
Questa funzione valuta le prestazioni del modello usando l'accuracy
@param trainer  il trainer del modello fine-tunato
@param test     il test set su cui testare il modello
'''
def test_model(trainer,test):
    metrics = trainer.evaluate(test)
    trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    return metrics


def ask_model(path):
    device = torch.device("cuda")
    classifier = pipeline("image-classification", model="sardegna-vit", device=device)
    image = Image.open(path)
    return classifier(image)

def show_prediction(path,pred,ground_truth):
    img = np.asarray(Image.open(path))
    imgplot = plt.imshow(img)
    plt.text(40, 40, pred[0]['label'])
    plt.text(200, 40, pred[0]['label'], color='green')
    plt.show()

def k_fold_cross():
    dataset = load_dataset("imagefolder", data_dir= "C:/Users/mocci/Desktop/MOST/Sardegna180/dataset-Sardegna180", split='train') # carico i dati
# Perform k-fold cross-validation
    num_folds = 5
    fold_results = []
    fold_datasets = []
    for fold_index in range(num_folds):
        fold_datasets.append(dataset.train_test_split(test_size=1/num_folds, seed=fold_index))

    for fold_index in range(num_folds):
        train_dataset = fold_datasets[fold_index]['train'].with_transform(transform)
        eval_dataset = fold_datasets[fold_index]['test'].with_transform(transform)
        trainer = create_trainer(train_dataset,eval_dataset,1)
        trainer = train_model(trainer)
        fold_result = trainer.evaluate()
        fold_results.append(fold_result)
        print(f"{fold_index} : {fold_results}")
    # Aggregate evaluation metrics across all folds
    return fold_results

# FUNZIONI DI SUPPORTO


'''
Questo transform prende un batch di esempi e ne estrae le feature chiamando il processor del modello
'''
def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs

'''
Questa funzione definisce come aggregare un insieme di istanze in un unico batch.
'pixel_values' sono le feature estratte, mentre 'labels' sono le etichette di classe
'''
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

'''
Questa funzione prende un'insieme di predizioni p e ne calcola l'accuracy
'''
def compute_metrics(p):
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("confusion_matrix")
    metric3 = evaluate.load("mse")
    metric4 = evaluate.load("precision")
    metric5 = evaluate.load("recall")

    accuracy = round(metric1.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["accuracy"],3)
    confusion_matrix = metric2.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["confusion_matrix"].tolist()
    mse = round(metric3.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["mse"],3)
    precision = round(metric4.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='macro')["precision"],3)
    recall = round(metric5.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='macro')["recall"],3)

    total = len(p.predictions)
    acc = 0
    for i in range(5):
            acc += confusion_matrix[i][i]
            if i > 0:
                acc += confusion_matrix[i][i-1]
            if i < 4:
                acc += confusion_matrix[i][i+1]
    one_out = round(acc / total,3)
    result = "\n"
    for i in range(5):
        result += str(confusion_matrix[i]) + "\n"
    


    return {"accuracy": accuracy, "mse": mse, "precision":precision, "recall": recall, "one_out": one_out} #"confusion_matrix":result}

def get_filenames_and_classes(folder):
    colnames = ['osm_id','osm_id_v0','osm_id_v1','filename','class']
    df = pd.read_csv(folder + '/data.csv', sep=';')
    df.columns = colnames
    file_names = [name for name in df['filename']]
    classes = [c for c in df['class']]
    return file_names, classes

def get_all_images_of_folder(path, nMax):
    imgs = []
    i = 0
    for f in os.listdir(path):
        if i < nMax:
            imgs.append(Image.open(os.path.join(path,f)))
        i +=1
    return imgs

def push_model_to_hub():
    model = AutoModelForImageClassification.from_pretrained("sardegna-vit")
    model.push_to_hub("Aenigmista/Sardegna-ViT")

def get_lat_lon_dataset():
    G = ox.graph_from_place('Sassari', network_type='drive')
    G_proj = ox.project_graph(G)
    df = pd.read_csv('dataset-Sardegna180/data180.csv')
    lat_lon = pd.DataFrame(columns=['lon','lat','heading','score'])
    for i in range(len(df)):
        entry = df.iloc[i]
        name = entry['name']
        parts = name.split('_')
        if parts[-1] == '1':
            lat = parts[1]
            lon = parts[2]
            score = entry['score']
            ne = ox.nearest_edges(G_proj, X=float(lat), Y=float(lon))
            p0 = G.nodes[ne[0]]
            p1 = G.nodes[ne[1]]
            heading = math.atan2(p1['y'] - p0['y'], p1['x'] - p0['x'])
        lat_lon = lat_lon._append({'lon':lon, 'lat':lat, 'heading':heading, 'score':score}, ignore_index=True)
    lat_lon.to_csv(f'lonlatSardegna.csv')
