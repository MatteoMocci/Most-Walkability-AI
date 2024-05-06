import torch
import numpy as np
from datasets import load_dataset, DatasetDict, load_metric
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments, pipeline
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import huggingface_hub

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
def create_trainer(train,valid,n,to_train):
    if to_train:
        model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=5, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForImageClassification.from_pretrained("AEnigmista/Sardegna-ViT", num_labels=5, ignore_mismatched_sizes=True)

    training_args = TrainingArguments(
    output_dir="./sardegna-vit",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    num_train_epochs=n,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=100,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
    use_cpu=False
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
    trainer.save_metrics("train", train_results.metrics)
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
    trainer.save_metrics("eval", metrics)     
    pass


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
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

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
    huggingface_hub.login("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")
    model = AutoModelForImageClassification.from_pretrained("sardegna-vit")
    model.push_to_hub("Aenigmista/Sardegna-ViT")