import torch
import torch.nn as nn
import torchvision

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)  
        key = self.key(x).permute(0, 2, 1)  
        value = self.value(x)  

        attention_scores = torch.matmul(query, key)  
        attention_weights = self.softmax(attention_scores)  
        out = torch.matmul(attention_weights, value)  
        return out + x



class SelfAttentionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  
        self.attention = SelfAttention(2048)  
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.resnet(x)  
        x = x.unsqueeze(1)  
        x = self.attention(x)  
        x = x.squeeze(1)  
        x = self.classifier(x)  
        return x


class MultiHeadAttentionCNN(nn.Module):
    def __init__(self, num_classes=2, embed_dim=1000, num_heads=8, dropout=0.2):
        super().__init__()
        self.resnet = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  
        
        
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        x = self.resnet(x)  
        x = x.unsqueeze(1)  
        
        x, _ = self.multihead_attention(x, x, x)  
        x = x.squeeze(1) 
        x = self.classifier(x)
        return x

        