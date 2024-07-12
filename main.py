import sardegna_scripts as sard
import itertools
import time
import os

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

from transformers import AutoModelForImageClassification
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchviz import make_dot

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
        for batch in tqdm(dataloader, desc=f"Extracting features for {save_path}"):
            inputs = batch['pixel_values'].to(device)
            outputs = model(inputs, output_hidden_states=True)
            feature = outputs.hidden_states[-1][:, 0, :]  # [batch_size, sequence_length, hidden_size] -> [batch_size, hidden_size]
            features.append(feature.cpu().numpy())
            # Print the shape of the embeddings from the hidden layers
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
    
    # Define the bins for feature importance
    bins = np.linspace(0, max(coefficients), 10)
    
    # Count the number of features in each bin
    hist, bin_edges = np.histogram(coefficients, bins=bins)

    # Get the threshold used by the feature selector
    threshold = np.mean(np.abs(rf.coef_))
    print(f"Feature selection threshold: {threshold}")


    # Plot the histogram of feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", align="edge")
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Feature Selection Threshold')
    plt.xlabel('Feature Importance Range')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Importances')
    plt.show()

    np.save(save_path, X_new)
    return X_new, selector

'''
Questa funzione permette di stampare a video un istogramma che rappresenta la distribuzione delle feature selezionate. 
Nell'asse delle x è presente il valore di importanza, discretizzato in 10 range e nell'asse delle y il numero di feature che ha tale importanza.
Ogni barra dell'istogramma è divisa in 2: una parte blu che indica il numero di feature appartenenti al modello streetview e una arancione per il numero di feature appartenenti al modello satellitare
@param X                Le feature di training combinate
@param y                Le label delle istanze di training
@param num_features1    Il numero di features estratte dal primo modello, per poter dividere le feature concatenate in base al modello da cui sono esrtatte.
'''
def print_combined_feature_importance(X, y, num_features1):
    rf = LinearSVC(C=0.01, penalty="l2", dual=False).fit(X, y)
    # Extract feature coefficients
    coefficients = np.mean(np.abs(rf.coef_), axis=0)

    # Define the bins for feature importance
    bins = np.linspace(0, max(coefficients), 10)
    
    # Count the number of features in each bin for both models
    hist1, _ = np.histogram(coefficients[:num_features1], bins=bins)
    hist2, _ = np.histogram(coefficients[num_features1:], bins=bins)

    # Plot the stacked bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1], hist1, width=np.diff(bins), edgecolor="black", align="edge", label='Feature Modello Street-view', color='blue')
    plt.bar(bins[:-1], hist2, width=np.diff(bins), edgecolor="black", align="edge", bottom=hist1, label='Feature Modello Satellitare', color='orange')
    plt.xlabel('Feature Importance Range')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Importances')
    plt.legend()
    plt.show()

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
Questa classe rappresenta la rete neurale deep che effettua la classificazione finale delle feature.
Tre layer convoluzionali, un layer di dropout e un secondo layer fully connected che restituisce i logit per le classi.
'''
class CombinedModel(nn.Module):
    def __init__(self, feature_dim1, feature_dim2, num_classes):
        super(CombinedModel, self).__init__()
        self.combined_dim = feature_dim1 + feature_dim2

        # Define CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (self.combined_dim // 8), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)  # Assuming num_classes for classification
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, combined):
        x = combined.unsqueeze(1)  # Add a channel dimension
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
'''
Questo metodo effettua la classificazione finale delle feature raccolte da entrambe le immagini tramite Support Vector Machine con Kernel rbf.
@param train_combined           le feature selezionate di training per dati streetview e satellitari
@param valid_combined           le feature selezionate di validation per dati streetview e satellitari
@param test_combined            le feature selezionate di test per dati streetview e satellitari
@param train_street             le feature di training per i dati street-view (serve per ricavare le label)
@param valid_street             le feature di validation per i dati street-view (serve per ricavare le label)
@param test_street              le feature di test per i dati street-view (serve per ricavare le label)
@param train_street_selected    le feature selezionate dai dati streetview (serve per ricavare il numero di feature selezionate dal modello street-view)
@param train_sat_selected       le feature selezionate dai dati satellitari (serve per ricavare il numero di feature selezionate dal modello satellitare)
'''
def final_classification_svm(train_combined, valid_combined, test_combined, train_street, valid_street, test_street, train_street_selected, train_sat_selected):
    svm_model = SVC(kernel='rbf', degree=3, C=1.0, random_state=42, probability=True)                                                                              # Inizializza il modello SVM
    svm_model.fit(train_combined, train_street['label'])                                                                                                           # Fitta il modello

    print_combined_feature_importance(train_combined, train_street['label'],train_street_selected.shape[1])
    sard.test_svm_model(svm_model, valid_combined, test_combined, train_street, valid_street, test_street)                                                         # Validation e test del modello

'''
Questo metodo effettua la classificazione finale delle feature raccolte da entrambe le immagini tramite una rete deep torch.
@param train_combined           le feature selezionate di training per dati streetview e satellitari
@param valid_combined           le feature selezionate di validation per dati streetview e satellitari
@param test_combined            le feature selezionate di test per dati streetview e satellitari
@param train_street             le feature di training per i dati street-view (serve per ricavare le label)
@param valid_street             le feature di validation per i dati street-view (serve per ricavare le label)
@param test_street              le feature di test per i dati street-view (serve per ricavare le label)
@param train_street_selected    le feature selezionate dai dati streetview (serve per ricavare il numero di feature selezionate dal modello street-view)
@param train_sat_selected       le feature selezionate dai dati satellitari (serve per ricavare il numero di feature selezionate dal modello satellitare)
@param device                   cpu o GPU (cuda)
'''
def final_classification_deep(train_combined, valid_combined, test_combined, train_street, valid_street, test_street, train_street_selected, train_sat_selected,device):

    num_classes = 5                                                                                                             # Numero classi

   
    combined_model = CombinedModel(train_street_selected.shape[1], train_sat_selected.shape[1], num_classes).to(device)         # Creazione modello combinato e caricamento su GPU

    criterion = nn.CrossEntropyLoss()                                                                                           # Definizione loss e optimizer
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)

    print_combined_feature_importance(train_combined, train_street['label'],train_street_selected.shape[1])
    train_combined = torch.tensor(train_combined).to(device)                                                                    # Trasformazione dati training in tensore e caricamento su GPU


    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        combined_model.train()
        optimizer.zero_grad()
        outputs = combined_model(train_combined)
        loss = criterion(outputs, torch.tensor(train_street['label']).to(device))
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(combined_model.state_dict(), 'combined_model.pth')                                                               # Salvataggio modello
    sard.test_torch_model(combined_model, valid_combined,test_combined,train_street,valid_street,test_street,device)            # Valuta le performance del modello su Validation e test set

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
        '''
        trainer_street = sard.create_trainer(train_street,valid_street,n=10,checkpoint='sardegna-vit')
        print("Metriche validation set - Streetview:")
        print(sard.test_model(trainer_street, valid_street))
        print("Metriche test set - Streetview:")
        print(sard.test_model(trainer_street, test_street))
        '''

    else:                                                                                                                                                         # altrimenti
        trainer_street = sard.create_trainer(train_street,valid_street,n=10,lr=1e-4,optim='adamw_hf',output_dir='./sardegna-vit')                                 # crea il trainer
        trainer_street = sard.train_model(trainer_street)                                                                                                         # esegui l'addestramento su street-view
        metrics_street = sard.test_model(trainer_street, test_street)
        print(metrics_street)                                                                                                                                     # testa il modello
        street_model = AutoModelForImageClassification.from_pretrained("sardegna-vit").to(device)                                                                 # carica il modello appena addestrato

    if os.path.exists('./satellite-vit'):                                                                                                                         # Se il modello addestrato sulle immagini satellitari è presente
        sat_model = AutoModelForImageClassification.from_pretrained("satellite-vit").to(device)
        '''                                                                   # caricalo
        trainer_sat = sard.create_trainer(train_sat,valid_sat,n=10,checkpoint='satellite-vit')
        print("Metriche validation set - Satellite:")
        print(sard.test_model(trainer_sat, valid_sat))
        print("Metriche test set - Satellite:")
        print(sard.test_model(trainer_sat, test_sat))
        '''
                                                                           
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

    use_svm = False
    if use_svm:
        final_classification_svm(train_combined,valid_combined,test_combined,train_street,valid_street,test_street,train_street_selected,train_sat_selected)            # Effettua la classificazione finale con SVM
    else:
        final_classification_deep(train_combined,valid_combined,test_combined,train_street,valid_street,test_street,train_street_selected,train_sat_selected,device)    # Altrimenti usa una rete torch custom

if __name__ == '__main__':
    walkability_pipeline()