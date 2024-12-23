import torch
import numpy as np
from datasets import load_dataset, DatasetDict
import evaluate 
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments, pipeline
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import osmnx as ox
import math
import osmapi as osm
import requests
import shutil
from openpyxl import load_workbook
from collections import Counter
from sklearn.inspection import permutation_importance
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, log_loss, f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
import random
import dual_encoder as de

api = osm.OsmApi() 
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", ignore_mismatched_sizes=True)

'''
Questo script carica le immagini dalla cartella dataset-Sardegna180 e crea un dataset diviso in train (80%), validation (10%) e test (10%).
La label di ciascuna immagine è direttamente determinata dal nome della cartella a cui essa appartiene. La funzione restituisce i tre split dopo aver estratto le feature di ogni immagine
'''
def load_data(option='180', seed=42):
    if option == '180':
        dataset = load_dataset("imagefolder", data_dir= "C:/Users/mocci/Desktop/MOST/Sardegna180/dataset-Sardegna180", split='train').train_test_split(test_size = 0.2,seed=seed)
    elif option == 'satellite':
        dataset = load_dataset("imagefolder", data_dir= "C:/Users/mocci/Desktop/MOST/Sardegna180/satellite", split='train').train_test_split(test_size = 0.2,seed=seed)
    else:
        raise ValueError("Nessuna opzione specificata!")
    test_valid = dataset['test'].train_test_split(test_size=0.5, seed=seed)
    # gather everyone if you want to have a single DatasetDict
    dataset = DatasetDict({
        'train': dataset['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})
    # Apply transformations
    train = dataset['train'].map(transform, batched=True)
    valid = dataset['valid'].map(transform, batched=True)
    test = dataset['test'].map(transform, batched=True)
    return train, valid, test


def new_load_data(option='180'):
    if option == '180':
        train = de.StreetSatDataset('paired_datasets/train.npy',street=True,sat=False)
        valid = de.StreetSatDataset('paired_datasets/valid.npy',street=True,sat=False)
        test = de.StreetSatDataset('paired_datasets/test.npy',street=True,sat=False)
    elif option == 'satellite':
        train = de.StreetSatDataset('paired_datasets/train.npy',street=False,sat=True)
        valid = de.StreetSatDataset('paired_datasets/valid.npy',street=False,sat=True)
        test = de.StreetSatDataset('paired_datasets/test.npy',street=False,sat=True)
    return train,valid,test
'''
Questa funzione definisce come aggregare un insieme di istanze in un unico batch.
'pixel_values' sono le feature estratte, mentre 'labels' sono le etichette di classe
'''
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([torch.tensor(x['pixel_values']) for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


'''
Questa funzione setta i parametri del Trainer e lo inizializza.
@param train        il training set
@param valid        il validation set
@param n            il numero di epoche per cui effettuare il training
@param to_train     True se il trainer viene creato per addestrare il modello o False se è necessario caricare l'ultimo modello per fare dei test
@param optim        l'optimizer da utilizzare per addestrare il modello
@param batch_size   la dimensione dei batch (8, 16 o 32)
@param checkpoint   il checkpoint di huggingface da utilizzare come rete pre-addestrata
@return il trainer creato
'''
def create_trainer(train,valid,n,lr=2e-4,optim="adamw_torch",batch_size=16,checkpoint="google/vit-base-patch16-224",output_dir='./sardegna-vit',collator=collate_fn, freeze_n_layers=40):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Addestra la rete con il checkpoint scelto oppure prendi l'ultima versione dall'hub
    model = AutoModelForImageClassification.from_pretrained(checkpoint, num_labels=5, ignore_mismatched_sizes=True).to(device)   
    # Freeze the first n layers
    freeze = False
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

    print_trainable_parameters(model)

    # La lista di parametri per il training, alcuni di questi possono esseere modificati passando i parametri alla
    training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    num_train_epochs=n,
    bf16=True,
    tf32=True,
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
    optim=optim,
    )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=processor,
    )
    return trainer

'''
Questa funzione addestra il modello e lo salva
@param trainer il modello da addestrare
@return il modello addestrato
'''
def train_model(trainer):
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_state()
    return trainer

'''
Questa funzione valuta le prestazioni del modello usando l'accuracy
@param trainer  il trainer del modello fine-tunato
@param test     il test set su cui testare il modello
@return il valore delle metriche
'''
def test_model(trainer,test):
    metrics = trainer.evaluate(test)
    trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    return metrics

'''
Questa funzione permette di fare inferenza sul modello caricato nell'hub di hugging face 
@param path il path del file dove si trova l'immagine da classificare
@return la predizione del modello per quell'immagine
'''
def ask_model(path):
    device = torch.device("cuda")
    classifier = pipeline("image-classification", model="AEnigmista/Sardegna-ViT", device=device)
    image = Image.open(path)
    return classifier(image)

'''
Questa funzione di supporto permette di mostrare l'immagine su cui è stata fatta una predizione, la corrispondente predizione del modello e la classe effettiva dell'immagine
@param path             il path dell'immagine da mostrare
@param pred             la predizione del modello per quell'immagine
@param ground_truth     la classe effettiva dell'immagine
'''
def show_prediction(path,pred,ground_truth):
    img = np.asarray(Image.open(path))
    imgplot = plt.imshow(img)
    plt.text(40, 40, pred[0]['label'])
    plt.text(200, 40, pred[0]['label'], color='green')
    plt.show()

'''
Questa funzione effettua una k_fold cross-validation sul dataset Sardegna180. Il dataset viene diviso in k parti e per ciascuna iterazione una viene selezionata per effettuare
il test mentre le altre vengono usate per effettuare il training del modello.
@return i risultati delle metriche per ciascuna fold
'''
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


'''
Questo transform prende un batch di esempi e ne estrae le feature chiamando il processor del modello
'''
def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')
    
    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs

def inference_transform(images):
    # Use the processor to transform the images into tensors
    inputs = processor(images, return_tensors='pt')
    return inputs['pixel_values']  # Only return the pixel values for inference


'''
Questa funzione prende un'insieme di predizioni p e ne calcola l'accuracy, la matrice di confusione, il mean square error, la precisione, il recall e la one_off accuracy
'''
def compute_metrics(p):
    show_matrix = True
    # carico le metriche
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("confusion_matrix")
    metric3 = evaluate.load("mse")
    metric4 = evaluate.load("precision")
    metric5 = evaluate.load("recall")
    metric6 = evaluate.load("f1")

    # calcolo le metriche
    accuracy = round(metric1.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["accuracy"],3)
    confusion_matrix = metric2.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["confusion_matrix"].tolist()
    mse = round(metric3.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["mse"],3)
    precision = round(metric4.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='macro')["precision"],3)
    recall = round(metric5.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='macro')["recall"],3)
    f1 = round(metric6.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='macro')["f1"],3)

    total = len(p.predictions)
    acc = 0

    # Calcolo one_off accuracy: (numero di predizioni che si distanziano max 1 dalla label del ground truth) / totale
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

    if show_matrix:
        for i in range(5):
            print(confusion_matrix[i])
    
    return {"accuracy": accuracy, "mse": mse, "precision":precision, "recall": recall, "one_out": one_out, "f1": f1} #"confusion_matrix":result}

'''
Questa funzione restituisce i nomi dei file e le label di classe di un dataset salvato in un file csv
@param folder il path alla cartella dove è contenuto il csv "data.csv"
@return i nomi dei file e le label di classe di quei file, inserite in una lista
'''
def get_filenames_and_classes(folder):
    colnames = ['osm_id','osm_id_v0','osm_id_v1','filename','class']
    df = pd.read_csv(folder + '/data.csv', sep=';')
    df.columns = colnames
    file_names = [name for name in df['filename']]
    classes = [c for c in df['class']]
    return file_names, classes

'''
Questa funzione prende due immagini image1 e image2 e restituisce un'immagine unica (new_image) che contiene entrambe, mettendo image2 a destra di image1 
'''
def merge_two_images(image1, image2):
    image1_size = image1.size
    new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    return new_image

'''
Questa funzione effettua il "merge" di una cartella di immagini. Per ogni diverso nome di file, esistono 4 immagini collegate. Le prime 2 e le altre 2 vengono unite in un'unica foto usando la funzione "merge_two_images"
Le immagini verranno salvate in una nuova cartella con il suffisso "_merged"
@param il path della cartella dove si trovano le immagini da unire
'''
def merge_images_folder(folder):
    file_names, _ = get_filenames_and_classes(folder)                               
    for i in range(len(file_names)):
        images = []  
        for j in range(4):
            curr_filename = folder + '/' + file_names[i] + '_' + str(j) + '.jpg'
            images.append(Image.open(curr_filename))
        img1 = merge_two_images(images[0],images[1])
        img2 = merge_two_images(images[2],images[3])
        img1.save(folder + "_merged/" + file_names[i] + '_0.jpg')
        img2.save(folder + "_merged/" + file_names[i] + '_1.jpg')

'''
Questa funzione restituisce una lista con le immagini caricate da una cartella path
@param path,    il path della cartella da cui prendere le immagini
@param nMax,    il numero massimo di immagini da prelevare dalla cartella
@return la lista con tutte le immagini caricate
'''                           
def get_all_images_of_folder(path, nMax=0):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    imgs = []
    i = 0
    if nMax == 0:
        nMax = len(os.listdir(path))
    for f in os.listdir(path):
        if i < nMax:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                imgs.append(Image.open(os.path.join(path,f)))
        i +=1
    return imgs

'''
Questa funzione effettua il caricamento dell'ultimo modello salvato sulla hub di HuggingFace
'''
def push_model_to_hub():
    model = AutoModelForImageClassification.from_pretrained("sardegna-vit")
    model.push_to_hub("Aenigmista/Sardegna-ViT")

'''
Questa funzione restituisce i modelli pre-addestrati su immagini street-view e satellitari
'''
def get_models():
    model_street = AutoModelForImageClassification.from_pretrained("sardegna-vit")
    model_satellite = AutoModelForImageClassification.from_pretrained("satellite-vit")
    return model_street, model_satellite

'''
Questa funzione restituisce l'altezza s.l.m di un punto rappresentato come stringa "lat,lon" utilizzando l'api "Elevation" di Google Maps.
'''
def get_elevation(api_key, location):
    base_url = "https://maps.googleapis.com/maps/api/elevation/json"
    
    # Construct the API request URL for a single location
    url = f"{base_url}?locations={location}&key={api_key}"
    
    # Send the HTTP GET request
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        if 'results' in data and data['results']:
            # Extract elevation data from the response
            elevation = data['results'][0]['elevation']
            return elevation
        else:
            print("No elevation data found in API response.")
    else:
        print(f"Error: {response.status_code} - {response.text}")

'''
Questa funzione converte il dataset .csv di streetview in un dataset .csv per le immagini satellitari. 
Dai nomi dei file vengono ricavati i valori di longitudine, latitudine e camminabilità (label di classe).
Con l'api Elevation viene calcolato il valore dell'altezza (z).
Viene ricavata l'arco del grafo delle strade di Open Street Map per ottenere i punti estremi e calcolare la direzione della strada. 
Lo stesso punto viene ripetuto due volte, uno con la direzione calcolata e l'altro aggiungendo 180° a quest'ultima, per ottenere una vista "davanti" e "dietro"
'''
def get_lat_lon_dataset(path='dataset-Sardegna180/data180.csv',output_path='lonlatSardegna.csv',valdala=False):

    apiK = open('api.txt').read()                                           # Ottieni api key
    df = pd.read_csv(path,sep=';')                    # Trasforma il .csv del dataset in un Dataframe
    lat_lon = pd.DataFrame(columns=['lon','lat','z','heading','score'])     # Crea un dataset vuoto con le colonne che andranno riempite
    if valdala:
        list_pictures = os.listdir('C:/Users/mocci/Desktop/MOST/Val d\'Ala/dataset_merged')

    iter_item = list_pictures if valdala else (df)
    for i in range(len(iter_item)):                                                # Per ogni elemento del dataset

        # ricava il valore di classe e il nome del file scomponendolo nelle varie informazioni
        if valdala:
            val = iter_item[i].replace(".jpg","")
            parts = val.split('_')        
            score = 0
        else:
            entry = df.iloc[i]
            name = entry['name']
            score = entry['score']
            parts = name.split('_')


        if parts[-1] == '0':                            # Se si tratta della prima foto per quel punto, ricava latitudine e longitudine
            lat = float(parts[1])
            lon = float(parts[2])
            
            p = (lat,lon)
            G = ox.graph_from_point(p)                      # Genera il grafo relativo al punto
            ne = ox.nearest_edges(G,X=lon,Y=lat)            # Ottieni l'arco del grafo più vicino al punto, ossia la strada a cui il punto appartiene
            p0 = api.NodeGet(ne[0])                         # Ottieni i due punti che rappresentano gli estremi della strada. L'informazione ottenuta è il nodo dell'id, quindi per ottenere le informazioni sui punti è usata l'api OSM
            p1 = api.NodeGet(ne[1])
            elevation = 50
            #elevation = get_elevation(apiK,f"{lat},{lon}")  # calcolo altezza punto tramite api elevation

            # Calcolo angolo della strada
            lat1 = math.radians(p0['lat'])
            lon1 = math.radians(p0['lon'])
            lat2 = math.radians(p1['lat'])
            lon2 = math.radians(p1['lon'])
            delta_lon = lon2 - lon1
            x = math.sin(delta_lon) * math.cos(lat2)
            y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))
            heading = (math.degrees(math.atan2(x, y)) + 360) % 360
            heading = (heading - 90) % 360


            lat_lon = lat_lon._append({'lon':lon, 'lat':lat, 'z':elevation, 'heading':heading, 'score':score}, ignore_index=True)                   # Aggiungi le informazioni relative alla riga nel dataset
            print(f"Processing {i/2}/{len(df)/2}")
        else:                                                                                                                                       # Se non è la prima foto, allora le informazioni sono già state estratte nell'iterazione precedente
            lat_lon = lat_lon._append({'lon':lon, 'lat':lat, 'z':elevation, 'heading': (heading + 180) % 360, 'score':score}, ignore_index=True)    # Riscrivi lo stesso, ma aggiungi 180° all'orientamento della strada, per rappresentare la vista dietro.
    lat_lon.to_csv(output_path)

'''
Effettua l'addestramento di una rete chiamando le funzioni definite sopra. Viene creato il trainer con i valori dei parametri specificati, addestrata la rete e calcolate le metriche sul test set
@param n                il numero di epoche 
@param lr               il learning rate
@param opt              l'optimizer da utilizzare
@param batch_size       la dimensione dei batch (8, 16 o 32)
@param checkpoint       il checkpoint da utilizzare per la rete preaddestrata
'''
def training_step(n=20,lr=2e-4,opt='adamw_torch',batch_size=16,checkpoint='google/vit-base-patch16-224',output_dir='./sardegna-vit'):
    if output_dir == './sardegna-vit':
        train, valid, test = load_data()
    else:
        train, valid, test = load_data('satellite')                                                                 # Caricamento di training, validation e test set. Split: 80 - 10 - 10
    trainer = create_trainer(train,valid,n,lr=lr,optim=opt,batch_size=batch_size,checkpoint=checkpoint,output_dir=output_dir)         # Caricamento di modello transformer "google/vit-base-patch16-224" e creazione trainer, l'ultimo valore è il numero di epoche
    trainer = train_model(trainer)                                                                              # Addestramento modello
    metrics = test_model(trainer, test)                                                                         # Test modello
    return metrics

'''
Effettua l'inferenza su un'immagine usando le funzioni definite sopra. Se show è a true, viene anche mostrata l'immagine.
'''
def inference_step(path,ground_truth):
    show = False                                    # Se a true, fa un plot dell'immagine mostrando la classe predetta e il ground_truth
    pred = ask_model(path)                          # Chiede al modello di predirre la classe 
    if show:                                        
        show_prediction(path,pred,ground_truth)
    return pred[0]['label']                         # Restituisce la label della classe più probabile secondo il modello

'''
Permette di effettuare il test di una rete già addestrata
'''
def test_only_step():
    train, valid, test = load_data()
    trainer = create_trainer(train,valid,20,False)
    test_model(trainer,test)

# Function to get predictions from the model
def get_predictions(model, tokenizer, dataset, batch_size=32):
    model.eval()
    predictions = []
    true_labels = []
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).tolist()
            predictions.extend(preds)
            true_labels.extend(batch['labels'].tolist())
    
    return predictions, true_labels

'''
Dato un dizionario in cui per ogni path è presente la sua predizione, raggruppa le immagini in base al valore di classe.
NB: Richiede che le cartelle siano già state create in precedenza
'''
def move_by_prediction(prediction):
    for key in prediction:
        file_name = os.path.basename(key)
        shutil.copyfile(key, f'Predictions/{prediction[key]}/{file_name}')

'''
Questa funzione crea un subplot per andare a visualizzare in maniera più compatta le immagini relative a ciascuna classe predetta dal modello.
Per ogni classe viene salvato un file .png con alcune delle immagini di quella classe, organizzate in una griglia
@param n_rows,  il numero di righe della griglia
@param n_cols,  il numero di colonne della griglia
@param figsize, la dimensione della figure su cui creare la griglia di immagini
'''
def display_images_by_class(n_rows, n_cols, figsize=(16,16)):
    for i in range(5):
        imgs = get_all_images_of_folder(f'Predictions/LABEL_{i}', n_rows * n_cols)
        if len(imgs) < n_rows * n_cols:
            if len(imgs) % 3 == 0:
                fix_rows = math.ceil(len(imgs)/3)
                fix_cols = 3
            else:
                fix_rows = len(imgs) // 2
                fix_cols = 2
            fig, ax = plt.subplots(fix_rows,fix_cols,figsize=figsize)
        else:
            fig, ax = plt.subplots(n_rows,n_cols,figsize=figsize)
        for j in range(n_rows*n_cols):
            if j < len(imgs):
                if len(imgs) >= n_rows * n_cols:
                    ax[j%n_rows][j//n_rows].imshow(imgs[j], aspect='auto')
                    ax[j % n_rows][j // n_rows].axis('off')  # Rimuovi gli assi
                else:
                    ax[j%fix_rows][j//fix_rows].imshow(imgs[j])
                    ax[j % fix_rows][j // fix_rows].axis('off')  # Rimuovi gli assi
        fig.suptitle(f"LABEL_{i}", fontsize=16)  # Aggiungi il titolo
        plt.subplots_adjust(wspace=0, hspace=0)  # Unisci le immagini
        fig.show()
        plt.savefig(f'LABEL_{i}.png') 

'''
Questa funzione permette di rappresentare in un csv le predizioni del modello.
In ogni riga è segnalato l'id del nodo, la latitudine, la longitudine e la classe secondo il modello
NB: Le immagini devono essere state opportunamente sistemate in cartelle in base alla label di classe attraverso la funzione move_by_prediction più sopra
'''
def write_prediction_csv():
    d_nodes = {}
    for i in range(5):
        for f in os.listdir(f'Predictions/Label_{i}'):
            f = os.path.splitext(f)[0]
            list_info = f.split('_')
            if list_info[0] not in d_nodes:
                d_nodes[list_info[0]] = {'coord': (list_info[1], list_info[2]) , list_info[3]: i}
            else:
                d_nodes[list_info[0]][list_info[3]] = i

    for k in d_nodes:
        class_counts = [d_nodes[k][str(i)] for i in range(2)]
        class_node = max(set(class_counts))
        for i in range(2):
            del d_nodes[k][str(i)]
        d_nodes[k]['class'] = class_node

    walkability = pd.DataFrame(columns=['osmId','lat','lon','class'])
    for k in d_nodes:
        coords = d_nodes[k]['coord']
        walkability = walkability._append({'osmId': k, 'lat': coords[0], 'lon': coords[1], 'class': d_nodes[k]['class']}, ignore_index = True)
    walkability.to_csv(f'model_predictions.csv')

'''
Questa funzione salva i risultati ottenuti in un file excel, è chiamata dalla funzione che permette di testare diverse configurazioni di modelli
@param idx          l'indice dell'iterazione corrente, serve per inserire i risultati nella riga corretta
@param checkpoint   il checkpoint usato per l'addestramento
@param n_epoch      il numero di epoche in cui il modello è stato addestrato
@param lr           il valore di learning rate usato per l'addestramento del modello
@param optimizer    l'optimizer scelto per l'addestramento
@param batch_size   la dimensione del batch (8, 16 o 32)
@param metrics      le metriche ottenute sul test set (accuracy, mse, precision, recall, one-off)
@param elapsed      il tempo necessario per l'addestramento del modello (in secondi)
@param name         il nome del file excel in cui salvare questi dati
'''
def save_results_to_excel(idx, checkpoint, n_epoch, lr, optimizer, batch_size,metrics, elapsed, name):
        workbook = load_workbook(filename=name)
        sheet = workbook.active

        # inserisci le informazioni relative ai parametri dell'esperimento
        sheet[f"A{idx}"] = checkpoint
        sheet[f"B{idx}"] = n_epoch
        sheet[f"C{idx}"] = lr
        sheet[f"D{idx}"] = optimizer
        sheet[f"E{idx}"] = batch_size

        # Inserisci il tempo impiegato e i valori delle metriche calcolati sul test set
        sheet[f"F{idx}"] = round(metrics['eval_loss'], 3)
        sheet[f"G{idx}"] = metrics['eval_accuracy']
        sheet[f"H{idx}"] = metrics['eval_mse']
        sheet[f"I{idx}"] = metrics['eval_precision']
        sheet[f"J{idx}"] = metrics['eval_recall']
        sheet[f"K{idx}"] = metrics['eval_one_out']
        sheet[f"L{idx}"] = round(elapsed,1)

        workbook.save(filename=name)
'''
Questo metodo legge tutte le immagini di un path e le scala a una dimensione di 1280x640, le converte in RGB e le inserisce in una cartella corrispondente al punteggio di camminabilità
'''
def convert_and_move_sat_pictures():
    df = pd.read_csv('lonlatSardegna.csv')
    path = 'C:/Users/mocci/Documents/Unreal Projects/Cesium/Saved/Screenshots/WindowsEditor'
    i = 1
    for file in os.listdir(path):
        print(f"Processing {i}/{len(df)}")
        parts = file.split("_")
        idx = parts[0]
        score = int(df.iloc[int(idx)]['score'])
        img = Image.open(path + '/' + file)
        img = img.resize((1280,640))
        rgb_img = img.convert('RGB')
        rgb_img.save(os.path.splitext((f'satellite/{score}/{file}'))[0]+ '.jpg')
        i += 1
'''
Questo metodo stampa a video la distribuzione delle classi per il training set e le predizioni di un eventuale test set. L'idea è quella di capire il comportamento di un classificatore dummie
@param train_labels la lista con le label del training set
@param test_labels la lista con le label del test set
'''       
def get_class_distributions(train_labels, test_labels):
    # Get the class distribution in the training set
    train_class_distribution = Counter(train_labels)
    print("Training Set Class Distribution:")
    for class_label, count in sorted(train_class_distribution.items()):
        print(f"Class {class_label}: {round(count/len(train_labels)* 100, 3)} %")

    test_class_distribution = Counter(test_labels)
    print("Test Set Predictions Class Distribution:")
    for class_label, count in sorted(test_class_distribution.items()):
        print(f"Class {class_label}: {round(count/len(test_labels) * 100, 3)} %")

'''
Questo metodo restituisce il numero di parametri del modello che possono essere fine-tunati
@param model il modello di cui restituire il numero di parametri
'''
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

'''
Questa funzione prende il modello SVM addestrato sulle feature concatenate e stampa un grafico che indica l'importanza delle feature, basato su permutation importance, per evidenziare
il contributo che le feature satellitari e streetview danno al modello.
@param  svc_model               il modello SVM, kernel rbf fittato sulle feature concatenate
@param  concatenated_features   le feature concatenate di training del modello street_view e di quello satellitare
@param  labels                  le classi delle istanze di training
@param  num_features1           il numero di features estratte dal modello street-view
@param  num_features2           il numero di features estratte dal modello satellitare
@param  top_n                   il numero di istanze da visualizzare (70 di default)
@param  n_repeats               il numero di ripetizioni per l'estrazione della permutation importance
'''
def print_model_chart_from_features(svc_model, concatenated_features, labels, num_features1, num_features2, top_n=70, n_repeats=30):
    
    # Creazioni nomi parametri e assegnazione colore (blue per street-view, arancione per satellitare)
    feature_names = [f'model1_feature_{i}' for i in range(num_features1)] + [f'model2_feature_{i}' for i in range(num_features2)]
    feature_colors = ['blue'] * num_features1 + ['orange'] * num_features2
    
    # Calcolo Permutation Importance con scikit-learn
    if os.path.exists('importance_scores.csv'):
        importance_scores = pd.read_csv('importance_scores.csv', header=None).values.flatten()
    else:
        print("Computing permutation importance...")
        importance_scores = np.zeros(concatenated_features.shape[1])
        
    
        for _ in tqdm(range(n_repeats), desc="Permutation Importance Calculation"):
            perm_importance = permutation_importance(svc_model, concatenated_features, labels, n_repeats=1, random_state=42, n_jobs=-1)
            importances_mean = perm_importance.importances_mean
            importances_mean[importances_mean < 0] = 0
            importance_scores += importances_mean

        importance_scores /= n_repeats  # Average the importance scores
        pd.DataFrame(importance_scores).to_csv('importance_scores.csv', index=False, header=False)

    # Estrazione dei valori di permutation importance e ordinamento
    feature_indices = np.argsort(importance_scores)[::-1][:top_n]  # Indices of top N features
    top_importance_scores = importance_scores[feature_indices]
    top_feature_names = [feature_names[i] for i in feature_indices]
    top_feature_colors = [feature_colors[i] for i in feature_indices]

    # Calcolo importanza totale per ciascun modello
    total_importance_model1 = np.sum(importance_scores[:num_features1])
    total_importance_model2 = np.sum(importance_scores[num_features1:])

    # Plot dell'importanza delle feature
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(top_feature_names, top_importance_scores, color=top_feature_colors)
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.yscale('log')  # Use logarithmic scale
    plt.title('Relative Importance of Selected Features')
    
    ax.annotate(f'Modello Street-view : {num_features1}(Importanza: {total_importance_model1})', xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='center', fontsize=12, color='blue')
    ax.annotate(f'Modello Satellitare: {num_features2}(Importanza: {total_importance_model2})', xy=(0.5, 0.90), xycoords='axes fraction', ha='center', va='center', fontsize=12, color='orange')
    
    plt.show()

'''
Questo metodo restituisce il numero di layer di cui è composto il modello model in input
@param model il modello di cui contare i layer
@return il numero di layer del modello
'''
def get_number_of_layers(model):
    # Counts the number of layers in the model
    return sum(1 for _ in model.named_parameters())

'''
Questo metodo prende il modello model e un parametro n e congela i primi n layer del modello.
Il congelamento fa sì che l'aggiornamento del gradiente non si propaghi a questi layer per trasferire la conoscenza del modello pre-addestrato al modello fine-tuned
@param  model   il modello di cui congelare i layer
@param  n       il numero di layer da congelare
'''
def freeze_layers(model, n):
    layers_frozen = 0
    for name, param in model.named_parameters():
        if layers_frozen >= n:
            break
        param.requires_grad = False
        layers_frozen += 1
'''
Questo metodo permette di testare cosa avverrebbe se il modello verrebbe riaddestrato congelando n layer. Si parte da 0 e ad ogni iterazione aumenta di 20 il numero di layer congelati
Nel file freeze_results.txt sono salvati i risultati dei vari esperimenti.
@param output_dir se viene specificato sardegna-vit, allora il test è fatto sul modello streetview, altrimenti si tratta di satellite-vit e quindi il test è fatto sul modello satellitare
'''
def test_freezing_layers(output_dir='./sardegna-vit'):
    total_layers = get_number_of_layers(AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224'))
    f = open('freeze_results.txt', 'a')
    print(total_layers)
    if output_dir == './sardegna-vit':
        train_dataset, valid_dataset, test_dataset = load_data()
    else:
        train_dataset, valid_dataset, test_dataset = load_data('satellite')
    for freeze_n_layers in range(0, total_layers + 1, 20):  # Change step as needed
        print(f"Freezing {freeze_n_layers} layers")
        trainer_street = create_trainer(train_dataset, valid_dataset, n=2, freeze_n_layers=freeze_n_layers, lr=1e-4, optim='adamw_hf',output_dir=output_dir)
        trainer_street.train()
        metrics_street = trainer_street.evaluate(test_dataset)
        print(f"Metrics with {freeze_n_layers} layers frozen: {metrics_street}")
        f.write(f"\nMetrics with {freeze_n_layers} layers frozen: {metrics_street}")

'''
Questa funzione calcola tutte le metriche necessarie ai fini della valutazione del modello SVM finale tramite scikit-learn
Vengono calcolate :
- Accuracy
- Classification Report (Recall, Precision, f1 score)
- Matrice di Confusione
- MSE (Mean Squared Error)
- Loss logaritmica
- One off accuracy (il rapporto tra le predizioni del modello che si distanziano al massimo di 1 dal ground truth e il numero totale di predizioni)
@param y_pred le predizioni del modello
@param y_proba le probabilità delle predizioni del modello
@param groundtruth il vero valore delle istanze predette dal modello
'''
def get_scikit_metrics(y_pred,y_proba,groundtruth):
    result = {}
    # Calcola accuracy, recall, precision, f1 score
    valid_accuracy = accuracy_score(groundtruth, y_pred)
    result['eval_accuracy'] = valid_accuracy
    print(f'Accuracy: {valid_accuracy}')
    print(classification_report(groundtruth, y_pred))

    # Calcola matrice di confusione
    cm = confusion_matrix(groundtruth, y_pred)

    # Calcolo di MSE
    mse = mean_squared_error(groundtruth, y_pred)

    # Calcolo della loss
    loss = log_loss(groundtruth, y_proba)
    result['eval_loss'] = loss

    print("Confusion Matrix:")
    print(cm)

    # Calcolo one-off accuracy
    acc = 0
    for i in range(5):
            acc += cm[i][i]
            if i > 0:
                acc += cm[i][i-1]
            if i < 4:
                acc += cm[i][i+1]
    one_off = round(acc / len(y_pred),3)

    result['eval_one_out'] = one_off
    result['f1'] = f1_score(y_true=groundtruth,y_pred=y_pred)
    result['eval_precision'] = precision_score(y_true=groundtruth, y_pred=y_pred)
    result['eval_recall'] = recall_score(y_true=groundtruth, y_pred=y_pred)
    print("One Off Accuracy:", one_off)
    print("\nMean Squared Error:", mse)
    print("\nLog Loss:", loss)
    return result

'''
Questa funzione effettua la valutazione del modello SVM con Validation e Test set, richiamando la funzione get_scikit_metrics() definita sopra
@param svm_model        il modello da testare
@param valid_combined   le feature del validation set
@param test_combined    le feature del test set
@train_street           i dati del training set     (serve per le label)
@valid_street           i dati del validation set   (serve per le label)
@test_street            i dati del test set         (serve per le label)
'''
def test_svm_model(svm_model, valid_combined, test_combined, train_street, valid_street, test_street, train_street_labels, valid_street_labels, test_street_labels):
    y_valid_pred = svm_model.predict(valid_combined)                                                                                                               
    y_pred_proba = svm_model.predict_proba(valid_combined)

    all_classes = np.arange(len(np.unique(train_street_labels)))
    full_y_pred_proba = np.zeros((y_pred_proba.shape[0], len(all_classes)))

    for i, class_idx in enumerate(svm_model.classes_):
        full_y_pred_proba[:, class_idx] = y_pred_proba[:, i]

    print("---Validation Set---")
    get_scikit_metrics(y_valid_pred, full_y_pred_proba, valid_street_labels)
    
    
    y_test_pred = svm_model.predict(test_combined)                                                                                                               
    y_pred_proba = svm_model.predict_proba(test_combined)

    all_classes = np.arange(len(np.unique(train_street_labels)))
    full_y_pred_proba = np.zeros((y_pred_proba.shape[0], len(all_classes)))

    for i, class_idx in enumerate(svm_model.classes_):
        full_y_pred_proba[:, class_idx] = y_pred_proba[:, i]

    print("---Test Set---")
    get_scikit_metrics(y_test_pred, full_y_pred_proba, test_street_labels)


def test_torch_model(torch_model, valid_combined, test_combined, train_street, valid_street, test_street, device='cpu'):
    # Move the model to the specified device
    torch_model.to(device)
    torch_model.eval()

    # Convert data to tensors and move to the specified device
    valid_combined_tensor = torch.tensor(valid_combined, dtype=torch.float32).to(device)
    test_combined_tensor = torch.tensor(test_combined, dtype=torch.float32).to(device)

    # Determine the number of classes from the training labels
    num_classes = len(np.unique(train_street['label']))

    # For validation set
    with torch.no_grad():
        y_valid_pred = torch_model(valid_combined_tensor)
        y_valid_pred_proba = torch.softmax(y_valid_pred, dim=1).cpu().numpy()
        y_valid_pred = torch.argmax(y_valid_pred, dim=1).cpu().numpy()
    
    full_y_valid_pred_proba = np.zeros((y_valid_pred_proba.shape[0], num_classes))

    for i in range(num_classes):
        full_y_valid_pred_proba[:, i] = y_valid_pred_proba[:, i]

    print("---Validation Set---")
    valid_res = get_scikit_metrics(y_valid_pred, full_y_valid_pred_proba, valid_street['label'])

    # For test set
    with torch.no_grad():
        y_test_pred = torch_model(test_combined_tensor)
        y_test_pred_proba = torch.softmax(y_test_pred, dim=1).cpu().numpy()
        y_test_pred = torch.argmax(y_test_pred, dim=1).cpu().numpy()

    full_y_test_pred_proba = np.zeros((y_test_pred_proba.shape[0], num_classes))

    for i in range(num_classes):
        full_y_test_pred_proba[:, i] = y_test_pred_proba[:, i]

    print("---Test Set---")
    test_res = get_scikit_metrics(y_test_pred, full_y_test_pred_proba, test_street['label'])
    return valid_res, test_res


def convert_label(p):
    return int(p.split('_')[1]) + 1

'''
Questa funzione è stata creata il 28/11/24 per ovviare a questo problema. Per ogni punto ho 2 immagini streetview e 2 immagini satellitari. Voglio che lo split sia sempre lo stesso
e che ogni immagine streetview venga inclusa con la corrispondente satellitare. Questa funzione crea dei dict a partire dalle coordinate e le chiavi di questi dict vengono mescolate per 
splittare il dataset in 80-10-10. I nomi dei file vengono salvati in array .npy. L'array in output è una lista di coppie
'''
def combine_photos(street_root, sat_root):
        street_dict = {}
        sat_dict = {}
        pair_dict = {}
        for i in range(5):
            street_paths = os.listdir(street_root + f'/{i}')
            sat_paths = os.listdir(sat_root + f'/{i}')
            street_coords = [path.split("_") for path in street_paths]
            sat_coords = [path.split("_") for path in sat_paths]
            for row in street_coords:
                lon = row[2]
                lat = row[1]
                if (lon,lat) not in street_dict.keys():
                    street_dict[(lon,lat)] = [f"{i}/{row[0]}_{row[1]}_{row[2]}_{row[3]}"]
                else:
                    if len(street_dict[(lon,lat)]) < 2:
                        street_dict[(lon,lat)].append(f"{i}/{row[0]}_{row[1]}_{row[2]}_{row[3]}")
            
              

            for row in sat_coords:
                lon = row[2]
                lat = row[1]
                if (lon,lat) not in sat_dict:
                    sat_dict[(lon,lat)] = [f"{i}/{row[0]}_{row[1]}_{row[2]}_{row[3]}"]
                else:
                    if len(sat_dict[(lon,lat)]) < 2:
                        sat_dict[(lon,lat)].append(f"{i}/{row[0]}_{row[1]}_{row[2]}_{row[3]}")

            for key in street_dict.keys():
                street_pair = street_dict[key]
                sat_pair = sat_dict[key]
                pair_dict[key] = [(street_pair[0],sat_pair[1]),(street_pair[1],sat_pair[0])]

        print(len(street_dict))
        len_train = round(0.8 * len(pair_dict.keys()))
        val_test_train = round (0.1 * len(pair_dict.keys()))
        key_split = list(pair_dict.keys())
        random.Random(42).shuffle(key_split)
        train_keys = key_split[0:len_train]
        val_keys = key_split[len_train:len_train+val_test_train]
        test_keys = key_split[len_train+val_test_train:len(pair_dict.keys())]
        train_split = [pair_dict[k] for k in train_keys]
        train_split = [k for l in train_split for k in l]
        valid_split = [pair_dict[k] for k in val_keys]
        valid_split = [k for l in valid_split for k in l]
        test_split = [pair_dict[k] for k in test_keys]
        test_split = [k for l in test_split for k in l]

        np.save('paired_datasets/train.npy',train_split)
        np.save('paired_datasets/valid.npy',valid_split)
        np.save('paired_datasets/test.npy', test_split)

        for k,v in street_dict.items():
            assert len(v) == 2

        for k,v in sat_dict.items():
            assert len(v) == 2

if __name__ == '__main__':
    train, valid, test = new_load_data('180')
    train2, valid2, test2 = new_load_data('satellite')
    train[0]['image'].show()
    train2[0]['image'].show()