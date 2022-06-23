import jittor as jt
from jittor import nn
import numpy as np

# nn.Linear(in_dim, out_dim)表示全连接层
# in_dim：输入向量维度
# out_dim：输出向量维度
def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2))
    return layers

class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim, n_classes):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(*block((latent_dim + n_classes), 128, normalize=False), 
                                   *block(128, 256), 
                                   *block(256, 512), 
                                   *block(512, 1024), 
                                   nn.Linear(1024, int(np.prod(img_shape))), 
                                   nn.Tanh())

    def execute(self, noise, labels):
        gen_input = jt.contrib.concat((self.label_emb(labels), noise), dim=1)
        img = self.model(gen_input)
        # 将img从1024维向量变为32*32矩阵
        img = img.view((img.shape[0], * self.img_shape))
        return img

class Discriminator(nn.Module):

    def __init__(self, img_shape, latent_dim, n_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(nn.Linear((n_classes + int(np.prod(img_shape))), 512), 
                                   nn.LeakyReLU(0.2), 
                                   nn.Linear(512, 512), 
                                   nn.Dropout(0.4), 
                                   nn.LeakyReLU(0.2), 
                                   nn.Linear(512, 512), 
                                   nn.Dropout(0.4), 
                                   nn.LeakyReLU(0.2), 
                                   # TODO: 添加最后一个线性层，最终输出为一个实数
                                   nn.Linear(512,1)
                                   )

    def execute(self, img, labels):
        d_in = jt.contrib.concat((img.view((img.shape[0], (- 1))), self.label_embedding(labels)), dim=1)
        # TODO: 将d_in输入到模型中并返回计算结果
        score = self.model(d_in)
        return score