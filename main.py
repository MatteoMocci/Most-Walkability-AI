from sardegna_scripts import load_data, create_trainer, train_model, test_model, ask_model, show_prediction, get_filenames_and_classes, get_all_images_of_folder, push_model_to_hub
import shutil
import matplotlib.pyplot as plt
import json 
import os
import math 
import pandas as pd

def training_step():
    train, valid, test = load_data()                # Caricamento di training, validation e test set. Split: 80 - 10 - 10
    trainer = create_trainer(train,valid,20)        # Caricamento di modello transformer "google/vit-base-patch16-224" e creazione trainer, l'ultimo valore è il numero di epoche
    trainer = train_model(trainer)                  # Addestramento modello
    test_model(trainer, test)                       # Test modello

def inference_step(path, ground_truth):
    show = False                                    # Se a true, fa un plot dell'immagine mostrando la classe predetta e il ground_truth
    pred = ask_model(path)                          # Chiede al modello di predirre la classe 
    if show:                                        
        show_prediction(path,pred,ground_truth)
    return pred[0]['label']                         # Restituisce la label della classe più probabile secondo il modello

def test_only_step():
    pass


def inference_on_dataset(folder):
    file_names, classes = get_filenames_and_classes(folder)                                         # Legge il csv del dataset e ottiene i nomi di ciascuna immagine e la rispettiva classe
    predictions = {}                                                                                # Questo dizionario conterrà le predizioni del modello in coppie nome_file : classe_predetta
    for i in range(len(file_names)):
        print(f"Processing {i}/{len(file_names)}")                                                                # Per ogni nome di file
        for j in range(4):                                                                          # Ci sono 4 immagini associate
            curr_filename = folder + '/' + file_names[i] + '_' + str(j) + '.jpg'                    # Ricava il path aggiungendo la parte relativa alla cartella, il numero dell'immagine e l'estensione .jpg
            predictions[curr_filename] = inference_step(curr_filename, classes[i])   # Inserisci nel dizionario la predizione per quella immagine (una per ognuna delle 4 immagini associate a un nome)
    return predictions                                                                              # Restituisci il dizionario

def move_by_prediction(prediction):
    for key in prediction:
        file_name = os.path.basename(key)
        shutil.copyfile(key, f'Predictions/{prediction[key]}/{file_name}')


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
        class_counts = [d_nodes[k][str(i)] for i in range(4)]
        class_node = max(set(class_counts), key = class_counts.count)
        for i in range(4):
            del d_nodes[k][str(i)]
        d_nodes[k]['class'] = class_node

    walkability = pd.DataFrame(columns=['osmId','lat','lon','class'])
    for k in d_nodes:
        coords = d_nodes[k]['coord']
        walkability = walkability._append({'osmId': k, 'lat': coords[0], 'lon': coords[1], 'class': d_nodes[k]['class']}, ignore_index = True)
    walkability.to_csv(f'model_predictions.csv')


push_model_to_hub()