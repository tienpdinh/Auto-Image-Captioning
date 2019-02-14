import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.hidden2target = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeds = self.word_embeddings(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)
        lstm_out, hidden = self.lstm(inputs)
        outputs = self.hidden2target(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        for _ in range(max_len):
            outputs, states = self.lstm(inputs, states)
            outputs = self.hidden2target(outputs.squeeze(1))
            out = outputs.max(1)[1]
            output.append(out.item())
            inputs = self.word_embeddings(out).unsqueeze(1)
        return output