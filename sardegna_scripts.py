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
    train_testvalid = dataset['train'].train_test_split(test_size=0.2,seed=seed)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5,seed=seed)
    # gather everyone if you want to have a single DatasetDict
    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})
    # Apply transformations
    train = dataset['train'].map(transform, batched=True)
    valid = dataset['valid'].map(transform, batched=True)
    test = dataset['test'].map(transform, batched=True)
    return train, valid, test

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
def create_trainer(train,valid,n,to_train=True,lr=2e-4,optim="adamw_torch",batch_size=16,checkpoint="google/vit-base-patch16-224",output_dir='./sardegna-vit',collator=collate_fn):

    # Addestra la rete con il checkpoint scelto oppure prendi l'ultima versione dall'hub
    if to_train:
        model = AutoModelForImageClassification.from_pretrained(checkpoint, num_labels=5, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForImageClassification.from_pretrained("AEnigmista/Sardegna-ViT", num_labels=5, ignore_mismatched_sizes=True)

    # La lista di parametri per il training, alcuni di questi possono esseere modificati passando i parametri alla
    training_args = TrainingArguments(
    output_dir=output_dir,
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


'''
Questa funzione prende un'insieme di predizioni p e ne calcola l'accuracy, la matrice di confusione, il mean square error, la precisione, il recall e la one_off accuracy
'''
def compute_metrics(p):

    # carico le metriche
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("confusion_matrix")
    metric3 = evaluate.load("mse")
    metric4 = evaluate.load("precision")
    metric5 = evaluate.load("recall")

    # calcolo le metriche
    accuracy = round(metric1.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["accuracy"],3)
    confusion_matrix = metric2.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["confusion_matrix"].tolist()
    mse = round(metric3.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["mse"],3)
    precision = round(metric4.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='macro')["precision"],3)
    recall = round(metric5.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='macro')["recall"],3)

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
    
    return {"accuracy": accuracy, "mse": mse, "precision":precision, "recall": recall, "one_out": one_out} #"confusion_matrix":result}

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
def get_all_images_of_folder(path, nMax):
    imgs = []
    i = 0
    for f in os.listdir(path):
        if i < nMax:
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
def get_lat_lon_dataset():

    apiK = open('api.txt').read()                                           # Ottieni api key
    df = pd.read_csv('dataset-Sardegna180/data180.csv')                     # Trasforma il .csv del dataset in un Dataframe
    lat_lon = pd.DataFrame(columns=['lon','lat','z','heading','score'])     # Crea un dataset vuoto con le colonne che andranno riempite


    for i in range(len(df)):                                                # Per ogni elemento del dataset

        # ricava il valore di classe e il nome del file scomponendolo nelle varie informazioni
        entry = df.iloc[i]
        name = entry['name']
        score = entry['score']
        parts = name.split('_')


        if parts[-1] == '1':                            # Se si tratta della prima foto per quel punto, ricava latitudine e longitudine
            lat = float(parts[1])
            lon = float(parts[2])
            
            p = (lat,lon)
            G = ox.graph_from_point(p)                      # Genera il grafo relativo al punto
            ne = ox.nearest_edges(G,X=lon,Y=lat)            # Ottieni l'arco del grafo più vicino al punto, ossia la strada a cui il punto appartiene
            p0 = api.NodeGet(ne[0])                         # Ottieni i due punti che rappresentano gli estremi della strada. L'informazione ottenuta è il nodo dell'id, quindi per ottenere le informazioni sui punti è usata l'api OSM
            p1 = api.NodeGet(ne[1])
            elevation = get_elevation(apiK,f"{lat},{lon}")  # calcolo altezza punto tramite api elevation

            # Calcolo angolo della strada
            lat1 = math.radians(p0['lat'])
            lon1 = math.radians(p0['lon'])
            lat2 = math.radians(p1['lat'])
            lon2 = math.radians(p1['lon'])
            delta_lon = lon2 - lon1
            x = math.sin(delta_lon) * math.cos(lat2)
            y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))
            heading = (math.degrees(math.atan2(x, y)) + 360) % 360
            heading = heading - 90 % 360


            lat_lon = lat_lon._append({'lon':lon, 'lat':lat, 'z':elevation, 'heading':heading, 'score':score}, ignore_index=True)                   # Aggiungi le informazioni relative alla riga nel dataset
            print(f"Processing {i/2}/{len(df)/2}")
        else:                                                                                                                                       # Se non è la prima foto, allora le informazioni sono già state estratte nell'iterazione precedente
            lat_lon = lat_lon._append({'lon':lon, 'lat':lat, 'z':elevation, 'heading': (heading + 180) % 360, 'score':score}, ignore_index=True)    # Riscrivi lo stesso, ma aggiungi 180° all'orientamento della strada, per rappresentare la vista dietro.
    lat_lon.to_csv(f'lonlatSardegna.csv')

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
Effettua l'inferenza su un'immaggine usando le funzioni definite sopra. Se show è a true, viene anche mostrata l'immagine.
'''
def inference_step(path, ground_truth):
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
        
if __name__ == '__main__':
    convert_and_move_sat_pictures()