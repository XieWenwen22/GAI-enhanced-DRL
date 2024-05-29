from torch import nn
import torch
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        # 调用父类方法初始化模块的state
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(input_dim, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

        # 解码器 ： [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training model [b, batch_size, 1, 28, 28]
        :return: the result of our training model
        """
        batch_size = x.shape[0]  # 每一批含有的样本的个数
        # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
        # tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，
        # 返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
        x = x.view(batch_size, self.input_dim)  # 一行代表一个样本

        # encoder
        mu, log_var = self.encode(x)
        # reparameterization trick
        sampled_z = self.reparameterization(mu, log_var)
        # decoder
        x_hat = self.decode(sampled_z)
        # reshape
        # x_hat = x_hat.view(batch_size, 1, 28, 28)
        return x_hat, mu, log_var

    def encode(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # 这里的“*”是点乘的意思

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        h = F.relu(self.fc4(z))
        # x_hat = torch.sigmoid(self.fc5(h))  # 图片数值取值为[0,1]，不宜用ReLU
        # print("Decode:",self.fc5(h).shape)
        return self.fc5(h)

    def loss_function(self,x_hat,x,mu,log_var):
        BCE = F.mse_loss(x_hat, x)
        # print("BCE:",BCE)

        # 2. KL-divergence
        # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
        # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
        # KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
        KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # 3. total loss
        loss = BCE + KLD
        return loss, BCE, KLD


