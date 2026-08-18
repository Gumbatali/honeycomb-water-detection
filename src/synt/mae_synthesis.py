import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermalConditionalMAE(nn.Module):
    """
    Conditional Masked Autoencoder для 1D-профилей тепловизионного контроля.
    Архитектура маскирует случайные участки профиля и восстанавливает их 
    на основе видимого контекста и заданного условия (процента заполнения).
    """
    def __init__(self, input_dim, cond_dim=1, hidden_dim=512, mask_ratio=0.50):
        super().__init__()
        self.input_dim = input_dim
        self.mask_ratio = mask_ratio
        
        # Кодировщик: принимает замаскированный профиль и метку-условие
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Декодировщик: восстанавливает исходный профиль
        self.decoder = nn.Sequential(
            nn.Linear((hidden_dim // 2) + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, c):
        # 1. Генерируем бинарную маску (True - видимо, False - замаскировано)
        mask = torch.rand(x.shape).to(x.device) > self.mask_ratio
        
        # 2. Применяем маску: зануляем скрытые пиксели/точки
        x_masked = x * mask.float()
        
        # 3. Кодируем
        xc = torch.cat([x_masked, c], dim=1)
        encoded = self.encoder(xc)
        
        # 4. Декодируем
        zc = torch.cat([encoded, c], dim=1)
        x_reconstructed = self.decoder(zc)
        
        return x_reconstructed, mask

    def generate(self, x_base, c_target):
        """
        Метод для синтеза данных. 
        Берем реальный профиль (x_base), сильно его маскируем и просим 
        декодер восстановить кривую уже под НОВОЕ условие (c_target).
        """
        self.eval()
        with torch.no_grad():
            # На этапе генерации можно маскировать до 75% профиля, чтобы
            # заставить сеть генерировать "новую" кривую, опираясь лишь на форму и условие
            mask = torch.rand(x_base.shape).to(x_base.device) > 0.75
            x_masked = x_base * mask.float()
            
            xc = torch.cat([x_masked, c_target], dim=1)
            encoded = self.encoder(xc)
            
            zc = torch.cat([encoded, c_target], dim=1)
            synth_profiles = self.decoder(zc)
            
        return synth_profiles

def mae_loss_fn(x_reconstructed, x_original, mask):
    """
    Функция потерь MAE. 
    Считает ошибку восстановления (MSE), акцентируя внимание на замаскированных точках.
    """
    loss = F.mse_loss(x_reconstructed, x_original, reduction='none')
    # Считаем среднюю ошибку только по тем точкам, которые модель НЕ видела
    masked_loss = (loss * (~mask).float()).sum() / (~mask).float().sum()
    
    # Для стабильности добавляем небольшой вес ошибки видимых точек
    visible_loss = (loss * mask.float()).sum() / mask.float().sum()
    
    return masked_loss + 0.1 * visible_loss