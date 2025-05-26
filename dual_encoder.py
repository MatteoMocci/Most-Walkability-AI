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
            outputs = self.vit(street, output_attentions=True)
        else:
            outputs = self.vit(sat,output_attentions=True)
        features = outputs.last_hidden_state[:, :, :]
        attentions = outputs.attentions[-1]

        cls_attn = attentions[:, :, 0, :]
        cls_attn = cls_attn.mean(dim=1)
        cls_attn = cls_attn[:, 1:]
        #attentions = cls_attn.mean(dim=0)
        
        return features, cls_attn

class DualEncoderModel(nn.Module):
    def __init__(self, checkpoint, num_classes=5):
        super(DualEncoderModel, self).__init__()
        
        labels = [str(x + 1) for x in range(num_classes)]
        label2id = {c:idx for idx,c in enumerate(labels)}
        id2label = {idx:c for idx,c in enumerate(labels)}

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
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, street=None, sat=None, labels=None):
        
        #Estraggo solo i pixel_values, fondamentalmente mi servono solo quelli
        street_inputs = street
        sat_inputs = sat

        # Chiamo i due encoder e ottengo le feature
        features_street, attentions_street = self.street_encoder(street=street_inputs)
        features_sat, attentions_sat = self.sat_encoder(sat=sat_inputs)

        street_threshold = torch.quantile(attentions_street, q=50 / 100.0)
        street_selected_indices = (attentions_street >= street_threshold).nonzero(as_tuple=True)[0]

        sat_threshold = torch.quantile(attentions_sat, q=50 / 100.0)
        sat_selected_indices = (attentions_sat >= sat_threshold).nonzero(as_tuple=True)[0]

        street_top_indices = street_selected_indices.unsqueeze(0).expand(attentions_street.size(0), -1)
        street_top_indices_expanded = street_top_indices.unsqueeze(-1).expand(-1, -1, features_street.size(-1))
        features_street = torch.gather(features_street, 1, street_top_indices_expanded)

        sat_top_indices = sat_selected_indices.unsqueeze(0).expand(attentions_sat.size(0), -1)
        sat_top_indices_expanded = sat_top_indices.unsqueeze(-1).expand(-1, -1, features_sat.size(-1))
        features_sat = torch.gather(features_sat, 1, sat_top_indices_expanded)

        # Concateno le feature e chiamo il classificatore
        x = torch.cat((features_street, features_sat), dim=1)
        # Average along the dim1 dimension
        x = x.mean(dim=1)  # Resulting shape: [batch, dim2]
    
        logits = self.classifier(x)

        # Restituisco loss e logit
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


