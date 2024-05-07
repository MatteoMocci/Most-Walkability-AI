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
def create_trainer(train,valid,n,to_train=True):
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
    print(metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


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
        trainer = create_trainer(train_dataset,eval_dataset,20)
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

    accuracy = metric1.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["accuracy"]
    confusion_matrix = metric2.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["confusion_matrix"].tolist()
    mse = metric3.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["mse"]
    precision = metric4.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='macro')["precision"]
    recall = metric5.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='macro')["recall"]

    total = len(p.predictions)
    acc = 0
    for i in range(5):
            acc += confusion_matrix[i][i]
            if i > 0:
                acc += confusion_matrix[i][i-1]
            if i < 4:
                acc += confusion_matrix[i][i+1]
    one_out = acc / total
    result = "\n"
    for i in range(5):
        result += str(confusion_matrix[i]) + "\n"
    


    return {"accuracy": accuracy, "mse": mse, "precision":precision, "recall": recall, "one_out": one_out, "confusion_matrix":result}

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

def cross_validation_metric():
    res = {0 : {'eval_loss': 0.6987358331680298, 'eval_accuracy': 0.7065934065934066, 'eval_mse': 0.41318681318681316, 'eval_precision': 0.6453796734188554, 'eval_recall': 0.6401923110041741, 'eval_one_out': 0.9637362637362638, 
    'eval_confusion_matrix': 	[[322, 26, 8, 0, 0],[29, 80, 88, 1, 3],[8, 36, 722, 36, 25],[1, 8, 173, 64, 71],[0, 0, 12, 9, 98]], 'eval_runtime': 23.4043, 'eval_samples_per_second': 77.763, 'eval_steps_per_second': 9.742, 'epoch': 19.87},
    1:{'eval_loss': 0.7200667262077332, 'eval_accuracy': 0.7115384615384616, 'eval_mse': 0.40604395604395604, 'eval_precision': 0.7234868510296545, 'eval_recall': 0.5512085569690024, 'eval_one_out': 0.9626373626373627, 'eval_confusion_matrix': [[361, 34, 10, 0, 0],[44, 73, 86, 0, 0],[4, 29, 759, 16, 0],[1, 6, 218, 60, 1],[0, 1, 46, 29, 42]],
    'eval_runtime': 22.8301, 'eval_samples_per_second': 79.719, 'eval_steps_per_second': 9.987, 'epoch': 19.87},
    2:{'eval_loss': 0.6707126498222351, 'eval_accuracy': 0.7329670329670329, 'eval_mse': 0.34065934065934067, 'eval_precision': 0.7250528104566383, 'eval_recall': 0.6399011719397928, 'eval_one_out': 0.9763736263736263, 
    'eval_confusion_matrix': [[327, 21, 9, 0, 0],[33, 80, 111, 1, 0],[1, 18, 745, 54, 3],[1, 4, 174, 116, 21],[0, 0, 24, 11, 66]],'eval_runtime': 22.4266, 'eval_samples_per_second': 81.154, 'eval_steps_per_second': 10.166, 'epoch': 19.87},   
    3 : {'eval_loss': 0.657822847366333,'eval_accuracy': 0.7362637362637363,'eval_mse': 0.34285714285714286, 'eval_precision': 0.7053789908268498,'eval_recall': 0.708530816809447, 'eval_one_out': 0.9736263736263736, 'eval_confusion_matrix': [[317, 38, 3, 0, 0],
    [26, 121, 53, 6, 0],[7, 71, 633, 107, 11],[0, 12, 89, 169, 18],[0, 0, 9, 30, 100]],'eval_runtime': 23.1635, 'eval_samples_per_second': 78.572, 'eval_steps_per_second': 9.843, 'epoch': 19.87},
    4 : 
    {'eval_loss': 0.6412188410758972, 
    'eval_accuracy': 0.7236263736263736, 
    'eval_mse': 0.35, 
    'eval_precision': 0.7012402505568551, 
    'eval_recall': 0.6601925945737401, 
    'eval_one_out': 0.9791208791208791, 
    'eval_confusion_matrix': 
    [[330, 46, 3, 0, 0],[16, 120, 58, 2, 0],[3, 74, 666, 96, 3],[2, 16, 120, 135, 10],[0, 2, 7, 45, 66]],
    'eval_runtime': 23.0454, 'eval_samples_per_second': 78.974, 'eval_steps_per_second': 9.894, 'epoch': 19.87}}
    avg_res =  {}
    
    for k in res[0]:
        if k != "eval_confusion_matrix":
            avg_res[k] = (res[0][k] + res[1][k] + res[2][k] + res[3][k] + res[4][k]) / 5
    m = []
    for i in range(5):
        row = []
        for j in range(5):
            val = 0
            for k in range(5):
                val += res[k]['eval_confusion_matrix'][i][j]
            row.append(val)
        m.append(row)

    avg_res["eval_confusion_matrix"] = m
    return avg_res