import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoEncoder3D(nn.Module):
    def __init__(self, in_channels=1, cond_dim=1, latent_dim=128):
        super().__init__()
        # Вход: (Batch, 1, 300, 86, 86)
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=(5, 4, 4), stride=(2, 2, 2), padding=(2, 1, 1))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(5, 4, 4), stride=(2, 2, 2), padding=(2, 1, 1))
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(5, 3, 3), stride=(2, 2, 2), padding=(2, 1, 1))
        
        # Глобальный пулинг сжимает пространственно-временные фичи в вектор
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Скрытое пространство с учетом условия
        self.fc_mu = nn.Linear(64 + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(64 + cond_dim, latent_dim)

    def forward(self, x, c):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        
        h = self.adaptive_pool(h)
        h = h.view(h.size(0), -1) # Flatten -> (Batch, 64)
        
        # Конкатенация признаков видео и условия (например, 0.5 для 50% воды)
        hc = torch.cat([h, c], dim=1)
        return self.fc_mu(hc), self.fc_logvar(hc)


class VideoDecoder3D(nn.Module):
    def __init__(self, out_channels=1, cond_dim=1, latent_dim=128, target_frames=300):
        super().__init__()
        self.target_frames = target_frames
        
        self.fc = nn.Linear(latent_dim + cond_dim, 64 * (target_frames // 8) * 10 * 10)
        
        # Используем Upsample + Conv3d (Resize-Conv) чтобы избежать шахматных артефактов на видео
        self.conv1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        h = F.relu(self.fc(zc))
        
        # Возвращаем 3D структуру: (Batch, 64, T/8, 10, 10)
        h = h.view(h.size(0), 64, self.target_frames // 8, 10, 10)
        
        # Блок 1: Увеличиваем в 2 раза
        h = F.interpolate(h, scale_factor=2, mode='trilinear', align_corners=False)
        h = F.relu(self.conv1(h))
        
        # Блок 2: Увеличиваем в 2 раза
        h = F.interpolate(h, scale_factor=2, mode='trilinear', align_corners=False)
        h = F.relu(self.conv2(h))
        
        # Блок 3: Вытягиваем точно до финального размера (T, 86, 86)
        h = F.interpolate(h, size=(self.target_frames, 86, 86), mode='trilinear', align_corners=False)
        out_video = self.conv3(h)
        
        return out_video


class VideoConditionalVAE(nn.Module):
    """Главный класс 3D Генератора"""
    def __init__(self, target_frames=300, cond_dim=1, latent_dim=128):
        super().__init__()
        self.encoder = VideoEncoder3D(cond_dim=cond_dim, latent_dim=latent_dim)
        self.decoder = VideoDecoder3D(cond_dim=cond_dim, latent_dim=latent_dim, target_frames=target_frames)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, c)
        return x_hat, mu, logvar

    def generate(self, c_tensor, target_frames=300):
        """Инференс: генерация видео из шума под заданное условие"""
        self.eval()
        with torch.no_grad():
            num_samples = c_tensor.size(0)
            z = torch.randn(num_samples, self.encoder.fc_mu.out_features).to(c_tensor.device)
            synth_video = self.decoder(z, c_tensor)
        return synth_video