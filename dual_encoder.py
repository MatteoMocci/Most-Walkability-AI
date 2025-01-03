import torch
import torch.nn as nn
from transformers import AutoModel

'''
Questa classe rappresenta un encoder del modello. L'architettura prevede 2 encoder: uno per immagini satellitari
e uno per immagini streetview.
'''
class ImageEncoder(nn.Module):
    def __init__(self, id2label, label2id, checkpoint):
        super(ImageEncoder, self).__init__()
        self.vit = AutoModel.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
    
    '''
    Ciascun encoder prende o street o sat, a seconda dei dati passati. In output vengono restituite le feature.
    '''
    def forward(self, street=None, sat=None):
        if street is not None:
            outputs = self.vit(street)
        else:
            outputs = self.vit(sat)
        features = outputs.last_hidden_state[:, 0, :]
        return features

class DualEncoderModel(nn.Module):
    def __init__(self, id2label, label2id, checkpoint, num_classes=5):
        super(DualEncoderModel, self).__init__()

        # Il modello ha due encoder: uno sulle immagini streetview, l'altro sulle satellitari
        self.street_encoder = ImageEncoder(id2label=id2label, label2id=label2id, checkpoint=checkpoint)
        self.sat_encoder = ImageEncoder(id2label=id2label, label2id=label2id, checkpoint=checkpoint)
        hidden_size = self.street_encoder.vit.config.hidden_size # numero di embedding/feature in output
        
        # Stampa di debug per sapere quanti parametri sono addestrabili
        num_params = sum([p.numel() for p in self.street_encoder.parameters()])
        trainable_params = sum([p.numel() for p in self.street_encoder.parameters() if p.requires_grad])
        print(f"{num_params = :,} | {trainable_params = :,}")

        # Layer di classificazione
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, street=None, sat=None, labels=None):
        
        #Estraggo solo i pixel_values, fondamentalmente mi servono solo quelli
        street_inputs = street
        sat_inputs = sat

        # Chiamo i due encoder e ottengo le feature
        features_street = self.street_encoder(street=street_inputs)
        features_sat = self.sat_encoder(sat=sat_inputs)

        # Concateno le feature e chiamo il classificatore
        combined_features = torch.cat((features_street, features_sat), dim=1)
        logits = self.classifier(combined_features)

        # Restituisco loss e logit
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


