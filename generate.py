import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def generate_synthetic_data(model, data_module, num_samples=100):
    model.eval()
    z = torch.randn(num_samples, 16)
    x_recon = model.decode(z).cpu().numpy()

    # Split into continuous features and label logit
    cont = x_recon[:, :30]
    label_logit = x_recon[:, 30:]

    # Inverse-transform continuous features back to original scale
    cont = data_module.scaler.inverse_transform(cont)

    # Convert label logit to binary via sigmoid + threshold
    label = (1 / (1 + np.exp(-label_logit)) >= 0.5).astype(int)

    df = pd.DataFrame(cont, columns=data_module.feature_columns)
    df[data_module.target_column] = label
    return df
