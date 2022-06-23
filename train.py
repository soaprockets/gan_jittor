import jittor as jt
import jittor.nn as nn
# 导入MNIST数据集
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
import argparse
import os
import numpy as np
from PIL import Image
from json.tool import main

from CGAN import Generator,Discriminator
from data_utils import *


# 默认使用第一块GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 使用cuda或cpu
if jt.has_cuda:
    jt.flags.use_cuda = 1

# 参数选项
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default=True, action='store_true', help='train mode or eval mode')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
    opt = parser.parse_args()
    print(opt)
    return opt


def train(opt):
    # ----------
    #  数据准备
    # ----------
    transform_ = transform.Compose([
        transform.Resize(opt.img_size),
        transform.Gray(),
        transform.ImageNormalize(mean=[0.5], std=[0.5]),
    ])
    dataloader = MNIST(train=True, transform=transform_).set_attrs(batch_size=opt.batch_size, shuffle=True)
    
    # ----------
    #  模型定义
    # ----------
    generator = Generator(opt.img_shape, opt.latent_dim, opt.n_classes)
    discriminator = Discriminator(opt.img_shape, opt.latent_dim, opt.n_classes)
    optimizer_G = nn.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # 损失函数：平方误差
    # 调用方法：adversarial_loss(网络输出A, 分类标签B)
    # 计算结果：(A-B)^2
    adversarial_loss = nn.MSELoss()

    # ----------
    #  模型训练
    # ----------
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # 数据标签，valid=1表示真实的图片，fake=0表示生成的图片
            valid = jt.ones([batch_size, 1]).float32().stop_grad()
            fake = jt.zeros([batch_size, 1]).float32().stop_grad()

            # 真实图片及其类别
            real_imgs = jt.array(imgs)
            labels = jt.array(labels)

            # -----------------
            #  训练生成器
            # -----------------

            # 采样随机噪声和数字类别作为生成器输入
            z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32()
            gen_labels = jt.array(np.random.randint(0, opt.n_classes, batch_size)).float32()

            # 生成一组图片
            gen_imgs = generator(z, gen_labels)
            # 损失函数衡量生成器欺骗判别器的能力，即希望判别器将生成图片分类为valid
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            g_loss.sync()
            optimizer_G.step(g_loss)

            # ---------------------
            #  训练判别器
            # ---------------------

            validity_real = discriminator(real_imgs, labels)
            """TODO: 计算真实类别的损失函数"""
            d_real_loss = adversarial_loss(validity_real, valid)

            validity_fake = discriminator(gen_imgs.stop_grad(), gen_labels)
            """TODO: 计算虚假类别的损失函数"""
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # 总的判别器损失
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.sync()
            optimizer_D.step(d_loss)
            
            if i  % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.data, g_loss.data)
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(generator, opt.latent_dim, n_row=10, batches_done=batches_done)

        if epoch % 10 == 0:
            generator.save("generator_last.pkl")
            discriminator.save("discriminator_last.pkl")


def test(opt):
    generator = Generator(opt.img_shape, opt.latent_dim, opt.n_classes)
    generator.load('generator_last.pkl')
    generator.eval()

    discriminator = Discriminator(opt.img_shape, opt.latent_dim, opt.n_classes)
    discriminator.load('discriminator_last.pkl')
    discriminator.eval()
    
    #TODO: 写入你注册时绑定的手机号（字符串类型）
    number = ""
    n_row = len(number)
    z = jt.array(np.random.normal(0, 1, (n_row, opt.latent_dim))).float32().stop_grad()
    labels = jt.array(np.array([int(number[num]) for num in range(n_row)])).float32().stop_grad()
    gen_imgs = generator(z,labels)

    img_array = gen_imgs.data.transpose((1,2,0,3))[0].reshape((gen_imgs.shape[2], -1))
    min_=img_array.min()
    max_=img_array.max()
    img_array=(img_array-min_)/(max_-min_)*255
    Image.fromarray(np.uint8(img_array)).save("result.png")


if __name__ == '__main__':
    opt = options()
    # 输入输出图像大小
    opt.img_shape = (opt.channels, opt.img_size, opt.img_size)
    if not opt.eval:
        # 训练
        train(opt)
    else:
        test(opt)


