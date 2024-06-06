import sardegna_scripts as sard

import itertools
import time
import os


from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

from transformers import AutoModelForImageClassification
import torch
from torch.utils.data import DataLoader
import numpy as np

'''
Questa funzione permette di testare diverse configurazioni di modelli specificando una lista di checkpoint, di epoche, di learning rate, di batch_size e di ottimizzatori
Viene generata una lista con tutte le possibili combinazioni di questi valori e per ciascuna combinazione viene addestrato un modello, calcolate le metriche sul test set e i dati vengono salvati su un file excel
'''
def test_configurations():

    checkpoints = ['google/vit-base-patch16-224','timm/efficientnet_b3.ra2_in1k',
                   'timm/resnet18.a1_in1k','microsoft/beit-base-patch16-224-pt22k-ft22k','timm/resnet50.a1_in1k']
    n_epochs = [10,15]
    lrs = [1e-4]
    batch_sizes = [16,32]
    optim = ['adamw_hf', 'adafactor']

    hyperparam_combinations = list(itertools.product(checkpoints, n_epochs, lrs, batch_sizes, optim))

    # Experiment loop
    for idx, (checkpoint, n_epoch, lr, batch_size, optimizer) in enumerate(hyperparam_combinations, start=2):
        print(f"Config num {idx-1}/{len(hyperparam_combinations)}")
        start = time.time()
        metrics = sard.training_step(n=n_epoch,lr=lr,opt=optimizer,batch_size=batch_size)
        end = time.time()
        elapsed = end - start
        sard.save_results_to_excel(idx=idx,checkpoint=checkpoint,n_epoch=n_epoch,lr=lr,optimizer=optimizer,batch_size=batch_size,metrics=metrics,elapsed=elapsed,name='best_results.xlsx')

'''
Questa funzione permette l'estrazione delle feature per il modello model sui dati contenuti in dataloader. Queste feature vengono salvate in "save_path"
@param model        il modello pre-addestrato che verrà usato per l'estrzione degli embedding
@param dataloader   il dataloader contenente i batch delle immagini da cui estrarre le feature
@param device       cpu o, se CUDA è disponibile, gpu
@param save_path    il path in cui salvare le feature estratte
@return le feature estratte
'''
def extract_features(model, dataloader, device, save_path):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device)
            outputs = model(inputs)
            feature = outputs[0]
            features.append(feature.cpu().numpy())
    features = np.concatenate(features, axis=0)
    np.save(save_path, features)
    return features

'''
Questa funzione effettua il fit di un feature selector basato su Random Forest sulle feature di training.
Questo feature selector viene poi usato per effettuare feature selection sulle feature di training
@param X            le feature estratte dal modello sui dati di training
@param y            le label delle istanze di training
@param save_path    il path in cui salvare le feature selezionate
@return le feature di training selezionate e il feature selector da usare poi su feature di validation e test
'''
def fit_feature_selector(X, y, save_path):
    rf = LinearSVC(C=0.01, penalty="l2", dual=False).fit(X, y)
    selector = SelectFromModel(rf, prefit=True)
    X_new = selector.transform(X)
    print(f"Prima: {X.shape} \t Dopo:{X_new.shape}")

    # Extract feature coefficients
    coefficients = np.mean(np.abs(rf.coef_), axis=0)
    

    # Create a plot of the feature coefficients
    feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, coefficients)
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    infotype = 'streetview model' if '_street_' in save_path else 'satellite model'
    plt.title('Feature Importance (Coefficient Values) from Linear SVC for ' + infotype)
    plt.xticks(rotation=45)
    plt.show()

    np.save(save_path, X_new)
    return X_new, selector

'''
Questa funzione effettua il passaggio di feature selection per feature di validation e test.
@param selector     il feature selector fittato sulle feature di training tramite "fit_feature_selector()"
@param X            le feature estratte dal modello sui dati di validation o test
@param save_path    il path in cui salvare le feature selezionate
return le feature di X selezionate tramite il feature selector
'''
def select_features(selector, X, save_path):
    selected = selector.transform(X)
    np.save(save_path, selected)
    return selected

'''
Questa funzione effettua la concatenazione di due array numpy di feature X1 e X2, salva e restituisce un unico array numpy X_combined
@param  X1          il vettore di feature estratte dal train, validation o test delle immagini street-view
@param  X2          il vettore di feature estratte dal train, validation o test delle immagini satellitari
@param  save_path   il path dove salvare le feature concatenate
@return le feature di immagini street-view e satellitari concatenate
'''
def concatenate_features(X1, X2, save_path):
    # I due vettori devono avere lo stesso numero di righe per essere concatenati
    if X1.shape[0] != X2.shape[0]:
        raise ValueError("The number of samples in both datasets must be the same.")
    
    # Concatena lungo l'asse delle feature (axis=1)
    X_combined = np.concatenate((X1, X2), axis=1)
    
    # Salva le feature concatenate
    np.save(save_path, X_combined)
    
    return X_combined

'''
Questa funzione esegue tutto ciò che è richiesto per addestrare il modello di classificazione della camminabilità di punti basandosi su immagini delle strade street-view e satellitari:
    1) Addestra o carica il modello delle immagini street-view
    2) Addestra o carica il modello delle immagini satellitari
    3) Estrae o carica le feature per i dati di training, validation e test dai modelli ottenuti nello step 1 e 2
    4) Esegue il fitting di un SVM con kernel lineare e penalità norma L2 per le feature di training ottenute nello step 3, per entrambi i modelli o carica le feature selezionate già presenti, saltando lo step 5
    5) Esegui feature selection anche su validation e test con il selector fittato sul training
    6) Normalizza le feature con Robust Scaler
    7) Concatena le feature estratte dal modello street e dal modello satellitare per avere un unico vettore di feature per il training, uno per il validation e uno per il test o carica quelle già presenti
    8) Fitta un SVM sulle feature con kernel radial based di training ed effettua validation e test
'''
def walkability_pipeline():
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                                                                                         # Se CUDA è presente, usa la GPU 
    train_street, valid_street, test_street = sard.load_data()                                                                                                    # Carica i dataset per training, validation e test per le immagini street-view
    train_sat, valid_sat, test_sat = sard.load_data('satellite')                                                                                                  # Carica i dataset per training, validation e test per le immagini satellitari

    if os.path.exists('./sardegna-vit'):                                                                                                                          # Se il modello addestrato sulle immagini street view è presente
        street_model = AutoModelForImageClassification.from_pretrained("sardegna-vit").to(device)                                                                 # caricalo

    else:                                                                                                                                                         # altrimenti
        trainer_street = sard.create_trainer(train_street,valid_street,n=10,lr=1e-4,optim='adamw_hf',output_dir='./sardegna-vit')                                 # crea il trainer
        trainer_street = sard.train_model(trainer_street)                                                                                                         # esegui l'addestramento su street-view
        metrics_street = sard.test_model(trainer_street, test_street)
        print(metrics_street)                                                                                                                                     # testa il modello
        street_model = AutoModelForImageClassification.from_pretrained("sardegna-vit").to(device)                                                                 # carica il modello appena addestrato

    if os.path.exists('./satellite-vit'):                                                                                                                         # Se il modello addestrato sulle immagini satellitari è presente
        sat_model = AutoModelForImageClassification.from_pretrained("satellite-vit").to(device)                                                                   # caricalo
    else:                                                                                                                                                         # altrimenti
        trainer_sat = sard.create_trainer(train_sat,valid_sat,n=10,lr=1e-4,optim='adamw_hf',output_dir='./satellite-vit')                                         # crea il trainer
        trainer_sat = sard.train_model(trainer_sat)                                                                                                               # esegui l'addestramento su immagini satellitari
        metrics_sat = sard.test_model(trainer_sat, test_sat)                                                                                                      # testa il modello
        print(metrics_sat)
        sat_model = AutoModelForImageClassification.from_pretrained("satellite-vit").to(device)                                                                   # carica il modello appena addestrato
    
    if os.path.exists('./features/train_street_features.npy') and os.path.exists('./features/train_sat_features.npy'):                                            # Se sono presenti le feature di training
        train_street_features = np.load('features/train_street_features.npy')                                                                                     # carica feature di training, validation e test per immagini street-view e satellitari
        valid_street_features = np.load('features/valid_street_features.npy')
        test_street_features  = np.load('features/test_street_features.npy')

        train_sat_features = np.load('features/train_sat_features.npy')
        valid_sat_features = np.load('features/valid_sat_features.npy')
        test_sat_features  = np.load('features/test_sat_features.npy')    
    else:                                                                                                                                                         # altrimenti
        train_street_loader = DataLoader(train_street, batch_size=batch_size, shuffle=False, collate_fn=sard.collate_fn)                                          # crea i data loader per i dataset
        valid_street_loader = DataLoader(valid_street, batch_size=batch_size, shuffle=False, collate_fn=sard.collate_fn)
        test_street_loader = DataLoader(test_street, batch_size=batch_size, shuffle=False, collate_fn=sard.collate_fn)

        train_sat_loader = DataLoader(train_sat, batch_size=batch_size, shuffle=False, collate_fn=sard.collate_fn)
        valid_sat_loader = DataLoader(valid_sat, batch_size=batch_size, shuffle=False, collate_fn=sard.collate_fn)
        test_sat_loader = DataLoader(test_sat, batch_size=batch_size, shuffle=False, collate_fn=sard.collate_fn)

        train_street_features = extract_features(model=street_model,dataloader=train_street_loader, device=device,save_path='features/train_street_features.npy') # estrai le feature di training, validation e test per immagini satellitari
        valid_street_features = extract_features(model=street_model,dataloader=valid_street_loader, device=device,save_path='features/valid_street_features.npy')
        test_street_features = extract_features(model=street_model,dataloader=test_street_loader, device=device,save_path='features/test_street_features.npy')

        train_sat_features = extract_features(model=sat_model,dataloader=train_sat_loader, device=device,save_path='features/train_sat_features.npy')
        valid_sat_features = extract_features(model=sat_model,dataloader=valid_sat_loader, device=device,save_path='features/valid_sat_features.npy')
        test_sat_features = extract_features(model=sat_model,dataloader=test_sat_loader, device=device,save_path='features/test_sat_features.npy')
        
    if os.path.exists('./features/train_street_selected.npy') and os.path.exists('./features/train_sat_selected.npy'):                                           # Se sono presenti le feature selezionate di training, caricale
        train_street_selected = np.load('features/train_street_selected.npy')
        valid_street_selected = np.load('features/valid_street_selected.npy')
        test_street_selected = np.load('features/test_street_selected.npy')

        train_sat_selected = np.load('features/train_sat_selected.npy')
        valid_sat_selected = np.load('features/valid_sat_selected.npy')
        test_sat_selected = np.load('features/test_sat_selected.npy')
    else:
        train_street_selected, street_selector = fit_feature_selector(train_street_features,train_street['label'],'features/train_street_selected.npy')           # altrimenti fitta un feature selector per street e sat sui dati di training e seleziona feature di train, valid e test
        train_sat_selected, sat_selector = fit_feature_selector(train_sat_features, train_sat['label'],'features/train_sat_selected.npy')

        valid_street_selected = select_features(street_selector, valid_street_features, 'features/valid_street_selected.npy')
        test_street_selected = select_features(street_selector, test_street_features, 'features/test_street_selected.npy')

        valid_sat_selected = select_features(sat_selector, valid_sat_features, 'features/valid_sat_selected.npy')
        test_sat_selected = select_features(sat_selector, test_sat_features, 'features/test_sat_selected.npy')
    
   

    if os.path.exists("features/train_combined.npy"):                                                                                                              # Se sono presenti le feature concatenate di training
        train_combined = np.load("features/train_combined.npy")                                                                                                    # caricale per train, valid e test
        valid_combined = np.load("features/valid_combined.npy")                                                                                                          
        test_combined  = np.load("features/test_combined.npy")
    else:
        scaler = RobustScaler()                                                                                                                                    # normalizza feature                                                                                                                                   
        train_street_selected = scaler.fit_transform(train_street_selected)
        valid_street_selected = scaler.fit_transform(valid_street_selected)
        test_street_selected = scaler.fit_transform(test_street_selected)

        train_sat_selected = scaler.fit_transform(train_sat_selected)
        valid_sat_selected = scaler.fit_transform(valid_sat_selected)
        test_sat_selected = scaler.fit_transform(test_sat_selected)

        train_combined = concatenate_features(train_street_selected, train_sat_selected, "features/train_combined.npy")                                            # altrimenti concatena usando np.concatenate per train, valid e test
        valid_combined = concatenate_features(valid_street_selected, valid_sat_selected, "features/valid_combined.npy")                                                    
        test_combined = concatenate_features(test_street_selected, test_sat_selected, "features/test_combined.npy")

    svm_model = SVC(kernel='rbf', degree=3, C=1.0, random_state=42)                                                                                                # Inizializza il modello SVM
    svm_model.fit(train_combined, train_street['label'])                                                                                                           # Fitta il modello

    
    y_valid_pred = svm_model.predict(valid_combined)                                                                                                               # Validazione del modello
    valid_accuracy = accuracy_score(valid_street['label'], y_valid_pred)
    print(f'Validation Accuracy: {valid_accuracy}')
    print(classification_report(valid_street['label'], y_valid_pred))

   
    y_test_pred = svm_model.predict(test_combined)                                                                                                                 # Testa il modello
    test_accuracy = accuracy_score(test_street['label'], y_test_pred)
    print(f'Test Accuracy: {test_accuracy}')
    print(classification_report(test_street['label'], y_test_pred))


if __name__ == '__main__':
    walkability_pipeline()