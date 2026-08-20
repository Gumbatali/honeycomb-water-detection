import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, cond_dim=1, latent_dim=64):
        super().__init__()
        # Вход: 1 канал (температура) + 1 канал (условие, растянутое на весь кадр) = 2 канала
        self.conv1 = nn.Conv2d(1 + cond_dim, 32, kernel_size=4, stride=2, padding=1) # -> 43x43
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)           # -> 21x21
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)          # -> 11x11
        self.flatten = nn.Flatten()
        
        # 128 каналов * 11 * 11 (для входного патча ~86x86)
        self.fc_mu = nn.Linear(128 * 11 * 11, latent_dim)
        self.fc_logvar = nn.Linear(128 * 11 * 11, latent_dim)

    def forward(self, x, c):
        # Растягиваем скаляр условия до размера картинки и конкатенируем
        c_map = c.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        xc = torch.cat([x, c_map], dim=1)
        
        h = F.relu(self.conv1(xc))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.flatten(h)
        return self.fc_mu(h), self.fc_logvar(h)

class ConvDecoder(nn.Module):
    def __init__(self, cond_dim=1, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim + cond_dim, 128 * 11 * 11)
        
        # Транспонированные свертки для "разворачивания" картинки
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0) # -> 21x21
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1)  # -> 43x43
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=0)   # -> 86x86

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        h = F.relu(self.fc(zc))
        h = h.view(-1, 128, 11, 11) # Решейп обратно в feature map
        
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        # На выходе линейная активация (или Sigmoid, если данные нормализованы 0..1)
        out_image = self.deconv3(h) 
        return out_image

class ImageConditionalVAE(nn.Module):
    """
    Генератор 2D-кадров (термограмм) для заданного процента дефекта.
    """
    def __init__(self, cond_dim=1, latent_dim=64):
        super().__init__()
        self.encoder = ConvEncoder(cond_dim, latent_dim)
        self.decoder = ConvDecoder(cond_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, c)
        return x_hat, mu, logvar

    def generate(self, c_tensor):
        self.eval()
        with torch.no_grad():
            z = torch.randn(c_tensor.size(0), self.decoder.fc.in_features - c_tensor.size(1)).to(c_tensor.device)
            synth_images = self.decoder(z, c_tensor)
        return synth_images