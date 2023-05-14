from torch import nn, Tensor
from torchvision import models
import torch
from typing import List, Optional
import math



class ContinuousDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, batch_first, dim_feedforward):
        super(ContinuousDecoderLayer, self).__init__(d_model=d_model, nhead=nhead, batch_first=batch_first, dim_feedforward=dim_feedforward)
        self.return_hiddens = False
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None):
        x = tgt
        if self.norm_first:
            h_sa = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + h_sa
            h_mha = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + mha_sa
            x = x + self._ff_block(self.norm3(x))
        else:
            h_sa = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + h_sa)
            h_mha = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            x = self.norm2(x + h_mha)
            x = self.norm3(x + self._ff_block(x))
        if self.return_hiddens == True:
            return x, h_sa, h_mha, tgt
        return x
    

    def forward(self, tgt: Tensor, memory: Tensor, sa_history: Tensor, mha_history: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None):
        x = tgt

        if self.norm_first:
            h_sa = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + h_sa
            x_l = x[:, -1].unsqueeze(1)
            x = torch.cat((sa_history[:, :-1], x_l), dim=1)
            h_mha = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + mha_sa
            x_l = x[:, -1].unsqueeze(1)
            x = torch.cat((mha_history[:, :-1], x_l), dim=1)
            x = x + self._ff_block(self.norm3(x))
        else:
            h_sa = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + h_sa)
            h_mha = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            x = self.norm2(x + h_mha)
            x = self.norm3(x + self._ff_block(x))
        
        return x, h_sa, h_mha, tgt


class ContinuousDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer: ContinuousDecoderLayer, num_layers: int, norm=None):
        super(ContinuousDecoder, self).__init__(decoder_layer, num_layers, norm)
        self.return_hiddens = False
    
    def get_hiddens(self, mode: bool) -> None:
        self.return_hiddens = mode
        for l in self.layers:
            l.return_hiddens = mode
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt

        sa_history = []
        mha_history = []

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            
            if self.return_hiddens:
                output, sa, mha = output
                sa_history.append(sa)
                mha_history.append(mha)

        if self.norm is not None:
            output = self.norm(output)

        return output, sa_history, mha_history

    def forward(self, tgt: Tensor, memory: Tensor, sa_history: List[Tensor], mha_history: List[Tensor], tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt

        for i, mod in enumerate(self.layers):
            output, sa, mha = mod(output, memory, sa_history[i], mha_history[i], tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            
            sa_history[i][:,-1] = sa[:, -1]
            mha_history[i][:,-1] = mha[:, -1]

        if self.norm is not None:
            output = self.norm(output)

        return output, sa_history, mha_history


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Vgg16Decoder(nn.Module):
    def __init__(self, params_model):
        super(Vgg16Decoder, self).__init__()
        self.num_classes = params_model["num_classes"]
        self.dr_rate= params_model["dr_rate"]
        self.pretrained = params_model["pretrained"]
        self.freeze_conv = params_model["freeze_conv"]
        self.rnn_hidden_size = params_model["rnn_hidden_size"]
        self.rnn_num_layers = params_model["rnn_num_layers"]
        self.train_seq = params_model["train_seq"]
        self.test_motion = False
        self.use_dynamic = True
        self.shuffle_dynamic = False
        
#         baseModel = models.vgg16(pretrained=pretrained)
        #Using pretrained facenet
        if self.pretrained:
            baseModel = models.vgg16(num_classes=8749)
            baseModel.features = torch.nn.DataParallel(baseModel.features)
            model_checkpoint = torch.load('/home/administrator/experiments/familiarity/pretraining/vgg16/models/119.pth')
            baseModel.load_state_dict(model_checkpoint['state_dict'])
            if self.freeze_conv:
                for param in baseModel.parameters():
                    param.requires_grad = False
        else:
            baseModel = models.vgg16()
        
        num_features = baseModel.classifier[-1].in_features
        del baseModel.classifier[-1]
        self.baseModel = baseModel
        self.dropout= nn.Dropout(self.dr_rate)
        self.linear = nn.Linear(num_features, self.rnn_hidden_size)
        self.positional_encoder = PositionalEncoding(self.rnn_hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.rnn_hidden_size, nhead=8, batch_first=True, dim_feedforward=self.rnn_hidden_size)
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.rnn_num_layers)
        self.decoder = transformer_decoder
        self.classify = True
        self.classifier = nn.Linear(self.rnn_hidden_size, self.num_classes) # rnn_hidden_size
    
    def frame_forward(self, x, h=None, c=None):
        raise NotImplemented

    def replace_classifier(self, num_classes: int):
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.rnn_hidden_size, self.num_classes) # rnn_hidden_size
        if torch.cuda.is_available():
            self.classifier.cuda()

    def switch_classify(self):
        self.classify = not self.classify

    def classify(self, mode: bool):
        self.classify = mode

    def remove_static_features(self, feats: Tensor):
        id_mean = torch.mean(feats, dim=1)
        feats = feats - id_mean.unsqueeze(1)
        return feats
    
    def remove_dynamic_features(self, feats: Tensor):
        id_mean = torch.mean(feats, dim=1)
        id_mean = id_mean.unsqueeze(1)
        return id_mean.repeat(1, feats.shape[1],1)
        
    def shuffle_dynamic_features(self, feats: Tensor):
        id_mean = torch.mean(feats, dim=1).unsqueeze(1)
        dynamic_features = feats - id_mean
        feats = id_mean + dynamic_features[torch.randperm(id_mean.shape[0])]
        return feats

    def should_use_dynamic_features(self, mode: bool):
        self.use_dynamic = mode

    def switch_dynamic_features(self):
        self.should_use_dynamic_features(not self.use_dynamic)

    def should_use_static_features(self, mode: bool):
        self.test_motion = not mode

    def switch_static_features(self):
        self.should_use_static_features(not self.test_motion)

    def should_shuffle_dynamic_features(self, mode: bool):
        self.shuffle_dynamic = mode

    def switch_shuffle_dynamic_features(self):
        self.should_shuffle_dynamic_features(not self.shuffle_dynamic)

    def seq_forward(self, x):
        bs, ts, c, h, w = x.shape
        frame_features = []
        # print(x.shape)
        x = x.reshape((bs * ts, c, h, w))
        features = self.baseModel(x)
        projected = self.linear(features)
        frame_features = projected.reshape((bs, ts, -1))
        if self.test_motion:
            frame_features = self.remove_static_features(frame_features)
        elif self.shuffle_dynamic:
            frame_features = self.shuffle_dynamic_features(frame_features)
        elif not self.use_dynamic:
            frame_features = self.remove_dynamic_features(frame_features)
        frame_features = self.positional_encoder(frame_features)
        transformed = self.decoder(frame_features, frame_features, tgt_mask=(torch.triu(torch.ones(ts,ts), 1).cuda().bool()))
        transformed = self.dropout(transformed)
        return self.classifier(transformed[:, -1])
    
    def forward(self, x, h=None, c=None):
        if self.train_seq:
            return self.seq_forward(x)
        else:
            return self.frame_forward(x, h, c)
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class DecoderOnly(nn.Module):
    def __init__(self, params_model):
        super(DecoderOnly, self).__init__()
        self.num_features = params_model["dim"]
        self.dr_rate= params_model["dr_rate"]
        self.pretrained = params_model["pretrained"]
        self.freeze_conv = params_model["freeze_conv"]
        self.rnn_hidden_size = params_model["rnn_hidden_size"]
        self.rnn_num_layers = params_model["rnn_num_layers"]
        self.train_seq = params_model["train_seq"]
        self.test_motion = False
        self.shuffle_dynamic = False

        self.dropout = nn.Dropout(self.dr_rate)
        self.linear = nn.Linear(self.num_features, self.rnn_hidden_size)
        self.positional_encoder = PositionalEncoding(self.rnn_hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.rnn_hidden_size, nhead=8, batch_first=True, dim_feedforward=self.rnn_hidden_size)
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.rnn_num_layers)
        self.decoder = transformer_decoder
        self.classifier = nn.Linear(self.rnn_hidden_size, self.num_features) # rnn_hidden_size

    def forward(self, x):
        bs, ts, d = x.shape
        x = x.reshape((bs * ts, d))
        projected = self.linear(x)
        frame_features = projected.reshape((bs, ts, -1))
        frame_features = self.positional_encoder(frame_features)
        transformed = self.decoder(frame_features, frame_features, tgt_mask=(torch.triu(torch.ones(ts,ts), 1).cuda().bool()))
        transformed = self.dropout(transformed)
        return self.classifier(transformed[:, -1])