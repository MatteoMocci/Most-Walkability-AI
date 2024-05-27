import sardegna_scripts as sard
import itertools
import time

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

if __name__ == '__main__':
    test_configurations()