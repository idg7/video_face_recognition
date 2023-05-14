from torch import nn
class Vgg16Rnn(nn.Module):
    def __init__(self, params_model):
        super(Vgg16Rnn, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate= params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]
        self.train_seq = params_model["train_seq"]
        
        baseModel = models.vgg16(pretrained=pretrained)
        num_features = baseModel.classifier[-1].in_features
        del baseModel.classifier[-1]
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.GRU(num_features, rnn_hidden_size, rnn_num_layers) #LSTM
        self.classifier = nn.Linear(rnn_hidden_size, num_classes)
    
    def frame_forward(self, x, h=None, c=None):
        batch_size, channels, height, width = x.shape
        cnn_features = self.baseModel((x))
        if (h is None) and (c is None):
            out, (hn, cn) = self.rnn(cnn_features.unsqueeze(1))
        else:
            out, (hn, cn) = self.rnn(cnn_features.unsqueeze(1), (h, c))
        out = self.dropout(out[:,-1])
        out = self.classifier(out)
        return out, hn, cn
    
    def seq_forward(self, x):
        batch_size, time_series, channels, height, width = x.shape
        if time_series < 16:
            print(x.shape)
        frame_idx = 0
        cnn_features = self.baseModel((x[:,frame_idx]))
        out, (hn, cn) = self.rnn(cnn_features.unsqueeze(1))
        for frame_idx in range(1, time_series):
            y = self.baseModel((x[:,frame_idx]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.classifier(out) 
        return out
    
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