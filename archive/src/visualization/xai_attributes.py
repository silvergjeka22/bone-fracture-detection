import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def visualize_triplet(method_network, title, inputs, attribution_all):
    '''
    Visualize the actual image, the calculated map, the map overlayed on the original image.

    Parameters
    -------
    method_network : name of the method

    title : title of the plot

    inputs : batch of images 

    attribution_all : dict of the attributions


    '''

    
    fig, axes = plt.subplots(len(inputs), 3, figsize=(12, 24))

    for i in range(len(inputs)):
        img = inputs[i].mean(0).cpu().numpy()
        sal = np.abs(attribution_all.get(method_network)[i])
        sal = gaussian_filter(sal, sigma=1.5)
        
        # Normalizza 0-1
        sal = (sal - sal.min()) / (sal.max() - sal.min())
        
        # Colonna 1: immagine originale
        axes[i, 0].imshow(img, cmap='gray')
        # if i == 0: axes[i, 0].set_title('Original', fontsize=15)
        axes[i, 0].axis('off')
        
        # Colonna 2: solo saliency
        axes[i, 1].imshow(sal, cmap='hot')
        if i == 0: axes[i, 1].set_title(title, fontsize=15)
        axes[i, 1].axis('off')
        
        # Colonna 3: overlay
        axes[i, 2].imshow(img, cmap='gray')
        axes[i, 2].imshow(sal, cmap='hot', alpha=0.5,
                        vmin=np.percentile(sal, 80), vmax=sal.max())
        # axes[i, 2].set_title(f'Overlay - {labels[i]}', fontsize=8)
        axes[i, 2].axis('off')

    plt.tight_layout()


def visualize_triplet_input_x_gradient(method_network, title, inputs, attribution_all):
    fig, axes = plt.subplots(8, 3, figsize=(12, 24))
    method_attributions = attribution_all.get(method_network)

    for i in range(8):
        img = inputs[i].mean(0).cpu().numpy()
        sal = method_attributions[i]  # Mappa Input x Gradient (224x224)
        
        # 1. Input x Gradient può presentare picchi isolati molto intensi (rumore "salt and pepper").
        # Usiamo il 99.5° percentile assoluto per scalare l'immagine senza farci influenzare da outlier.
        abs_sal = np.abs(sal)
        v_max = np.percentile(abs_sal, 99.5)
        if v_max == 0: v_max = 1
        sal_clipped = np.clip(sal, -v_max, v_max)
        
        # Normalizziamo simmetricamente tra -1 e 1 per la gestione dei colori
        sal_norm = sal_clipped / v_max
        
        # 2. Soglia di rumore personalizzata. 
        # Lo sfondo è già parzialmente filtrato dalla moltiplicazione per l'input, 
        # ma eliminiamo il restante 80% dei pixel con i pesi più deboli sull'osso.
        threshold = np.percentile(abs_sal, 80)
        
        # 3. Inizializziamo l'overlay RGBA trasparente
        overlay_rgba = np.zeros((*sal.shape, 4))
        
        pos_mask = (sal > 0) & (abs_sal > threshold)
        neg_mask = (sal < 0) & (abs_sal > threshold)
        
        # 4. Colorazione pixel-level ad alto contrasto (Verde positivo, Rosso negativo)
        # Sfruttiamo un'opacità decisa (max 0.85) perché i dettagli pixel-level sono finissimi 
        # e non coprono l'anatomia sottostante.
        overlay_rgba[pos_mask, 1] = 1.0  # Canale Verde
        overlay_rgba[pos_mask, 3] = np.abs(sal_norm[pos_mask]) * 0.85
        
        overlay_rgba[neg_mask, 0] = 1.0  # Canale Rosso
        overlay_rgba[neg_mask, 3] = np.abs(sal_norm[neg_mask]) * 0.85
        
        # --- PLOTTING ---
        
        # Colonna 1: Immagine radiografica originale
        axes[i, 0].imshow(img, cmap='gray')
        if i == 0: axes[i, 0].set_title('Original', fontsize=15)
        axes[i, 0].axis('off')
        
        # Colonna 2: Mappa IxG Pura (Scala bwr)
        # Noterai come lo sfondo nero rimanga nativamente molto più pulito rispetto a IG o Saliency
        axes[i, 1].imshow(sal_clipped, cmap='bwr', vmin=-v_max, vmax=v_max)
        if i == 0: axes[i, 1].set_title(title, fontsize=15)
        axes[i, 1].axis('off')
        
        # Colonna 3: Overlay Chirurgico (Radiografia + Pixel IxG)
        # I bordi corticali dell'osso e le rime di frattura si illuminano matematicamente in verde/rosso
        axes[i, 2].imshow(img, cmap='gray')
        axes[i, 2].imshow(overlay_rgba)
        if i == 0: axes[i, 2].set_title('IxG Overlay (Pixel-level)', fontsize=15)
        axes[i, 2].axis('off')

    plt.tight_layout()



def visualize_triplet_ig(method_network, title, inputs, attribution_all):
    fig, axes = plt.subplots(8, 3, figsize=(12, 24))
    method_attributions = attribution_all.get(method_network)

    for i in range(8):
        img = inputs[i].mean(0).cpu().numpy()
        sal = method_attributions[i]  # Mappa IG pixel-level (224x224)
        
        # 1. Nel caso di IG, le attribuzioni possono avere picchi di rumore estremi.
        # Per una visualizzazione stabile, clipiamo i valori oltre il 99° percentile.
        abs_sal = np.abs(sal)
        v_max = np.percentile(abs_sal, 99)
        if v_max == 0: v_max = 1
        sal_clipped = np.clip(sal, -v_max, v_max)
        
        # Normalizziamo simmetricamente tra -1 e 1
        sal_norm = sal_clipped / v_max
        
        # 2. Impostiamo una soglia di rumore molto selettiva
        # IG illumina moltissimi pixel; filtriamo via l'85% dei pixel a basso impatto
        threshold = np.percentile(abs_sal, 85)
        
        # 3. Creiamo l'overlay RGBA
        overlay_rgba = np.zeros((*sal.shape, 4))
        
        pos_mask = (sal > 0) & (abs_sal > threshold)
        neg_mask = (sal < 0) & (abs_sal > threshold)
        
        # 4. Assegnazione colori standard per IG (Verde = +, Rosso = -)
        # Moduliamo l'opacità in base al peso normalizzato del singolo pixel
        overlay_rgba[pos_mask, 1] = 1.0  # Canale Verde
        overlay_rgba[pos_mask, 3] = np.abs(sal_norm[pos_mask]) * 0.8  # Alpha più alto per evidenziare i dettagli fini
        
        overlay_rgba[neg_mask, 0] = 1.0  # Canale Rosso
        overlay_rgba[neg_mask, 3] = np.abs(sal_norm[neg_mask]) * 0.8  # Alpha più alto
        
        # --- PLOTTING ---
        
        # Colonna 1: Immagine originale
        axes[i, 0].imshow(img, cmap='gray')
        if i == 0: axes[i, 0].set_title('Original', fontsize=15)
        axes[i, 0].axis('off')
        
        # Colonna 2: Solo Mappa IG (Scala divergente Bianco-Nero-Rosso o bwr)
        # Mostra i pixel positivi e negativi come filamenti ad alta risoluzione
        axes[i, 1].imshow(sal_clipped, cmap='bwr', vmin=-v_max, vmax=v_max)
        if i == 0: axes[i, 1].set_title(title, fontsize=15)
        axes[i, 1].axis('off')
        
        # Colonna 3: Overlay Chirurgico Pixel-Level
        # Vedrai i contorni critici delle ossa o delle fratture "accendersi" in verde/rosso
        axes[i, 2].imshow(img, cmap='gray')
        axes[i, 2].imshow(overlay_rgba)
        if i == 0: axes[i, 2].set_title('IG Overlay (Pixel-level)', fontsize=15)
        axes[i, 2].axis('off')

    plt.tight_layout()


def visualize_triplet_LIME(method_network, title, inputs, attribution_all):
    fig, axes = plt.subplots(8, 3, figsize=(12, 24))
    method_attributions = attribution_all.get(method_network)

    for i in range(8):
        img = inputs[i].mean(0).cpu().numpy()
        sal = method_attributions[i]
        
        # 1. Calcoliamo i valori assoluti per trovare una soglia di attivazione
        abs_sal = np.abs(sal)
        max_val = np.max(abs_sal) if np.max(abs_sal) > 0 else 1
        
        # Impostiamo una soglia: coloriamo solo i blocchi significativi
        # Ad esempio, escludiamo il 75% dei blocchi con i pesi più bassi
        threshold = np.percentile(abs_sal, 75) 
        
        # 2. Creiamo l'overlay RGBA vuoto (trasparente)
        overlay_rgba = np.zeros((*sal.shape, 4))
        
        # Maschere per i blocchi che superano la soglia
        pos_mask = (sal > 0) & (abs_sal > threshold)
        neg_mask = (sal < 0) & (abs_sal > threshold)
        
        # 3. Moduliamo l'opacità in modo dinamico in base all'importanza del blocco
        # Più il peso è alto, più il colore è intenso (max opacità = 0.6)
        if max_val > 0:
            overlay_rgba[pos_mask, 0] = 0.0                      # R
            overlay_rgba[pos_mask, 1] = 1.0                      # G
            overlay_rgba[pos_mask, 2] = 0.0                      # B
            overlay_rgba[pos_mask, 3] = (abs_sal[pos_mask] / max_val) * 0.6  # Alpha dinamico
            
            overlay_rgba[neg_mask, 0] = 1.0                      # R
            overlay_rgba[neg_mask, 1] = 0.0                      # G
            overlay_rgba[neg_mask, 2] = 0.0                      # B
            overlay_rgba[neg_mask, 3] = (abs_sal[neg_mask] / max_val) * 0.6  # Alpha dinamico

        # --- PLOTTING ---
        
        # Colonna 1: Immagine originale (Radiografia pulita)
        axes[i, 0].imshow(img, cmap='gray')
        if i == 0: axes[i, 0].set_title('Original', fontsize=15)
        axes[i, 0].axis('off')
        
        # Colonna 2: Griglia dei pesi LIME (Scala bwr originale, molto utile)
        v_max = max(abs(sal.min()), abs(sal.max())) if sal.any() else 1
        axes[i, 1].imshow(sal, cmap='bwr', vmin=-v_max, vmax=v_max)
        if i == 0: axes[i, 1].set_title(title, fontsize=15)
        axes[i, 1].axis('off')
        
        # Colonna 3: Nuovo Overlay Pulito (Senza bordi gialli, solo evidenziazioni sfumate)
        axes[i, 2].imshow(img, cmap='gray')
        axes[i, 2].imshow(overlay_rgba)
        if i == 0: axes[i, 2].set_title('LIME Overlay (Filtered)', fontsize=15)
        axes[i, 2].axis('off')

    plt.tight_layout()



def visualize_triplet_shap(method_network, title, inputs, attribution_all):
    fig, axes = plt.subplots(8, 3, figsize=(12, 24))
    method_attributions = attribution_all.get(method_network)

    for i in range(8):
        img = inputs[i].mean(0).cpu().numpy()
        sal = method_attributions[i]  # Valori di Shapley (224x224)
        
        # 1. Trova il valore massimo assoluto per centrare la scala simmetricamente
        abs_sal = np.abs(sal)
        max_val = np.max(abs_sal) if np.max(abs_sal) > 0 else 1
        
        # 2. Applica una soglia percentuale per eliminare il rumore dello sfondo nero
        # Taglia fuori il 75% dei quadrati con impatto minore o quasi nullo
        threshold = np.percentile(abs_sal, 75)
        
        # 3. Inizializza l'overlay RGBA trasparente
        overlay_rgba = np.zeros((*sal.shape, 4))
        
        # Maschere per i blocchi significativi sopra la soglia
        pos_mask = (sal > 0) & (abs_sal > threshold)
        neg_mask = (sal < 0) & (abs_sal > threshold)
        
        # 4. Assegna i colori secondo lo standard SHAP:
        # ROSSO per l'attribuzione positiva, BLU per l'attribuzione negativa
        if max_val > 0:
            # Rosso (R=1, G=0, B=0) con opacità proporzionale all'impatto
            overlay_rgba[pos_mask, 0] = 1.0
            overlay_rgba[pos_mask, 3] = (abs_sal[pos_mask] / max_val) * 0.6
            
            # Blu (R=0, G=0, B=1) con opacità proporzionale all'impatto
            overlay_rgba[neg_mask, 2] = 1.0
            overlay_rgba[neg_mask, 3] = (abs_sal[neg_mask] / max_val) * 0.6

        # --- PLOTTING ---
        
        # Colonna 1: Immagine originale (Radiografia pulita)
        axes[i, 0].imshow(img, cmap='gray')
        if i == 0: axes[i, 0].set_title('Original', fontsize=15)
        axes[i, 0].axis('off')
        
        # Colonna 2: Griglia SHAP Pura (Usa 'bwr' per mappare nativamente i valori)
        # Il blu indica decremento, il rosso incremento, il bianco neutralità
        axes[i, 1].imshow(sal, cmap='bwr', vmin=-max_val, vmax=max_val)
        if i == 0: axes[i, 1].set_title(title, fontsize=15)
        axes[i, 1].axis('off')
        
        # Colonna 3: Overlay Finale SHAP Filtrato
        # I blocchi si integrano sull'osso senza griglie gialle e senza toccare lo sfondo vuoto
        axes[i, 2].imshow(img, cmap='gray')
        axes[i, 2].imshow(overlay_rgba)
        if i == 0: axes[i, 2].set_title('SHAP Overlay (Red/Blue)', fontsize=15)
        axes[i, 2].axis('off')

    plt.tight_layout()



def visualize_triplet_occlusion(method_network, title, inputs, attribution_all):
    fig, axes = plt.subplots(8, 3, figsize=(12, 24))
    method_attributions = attribution_all.get(method_network)

    for i in range(8):
        img = inputs[i].mean(0).cpu().numpy()
        sal = method_attributions[i]  # Mappa originale 224x224
        
        # 1. Smoothing per eliminare artefatti da scacchiera
        sal_smooth = gaussian_filter(sal, sigma=2.0)
        
        # 2. Normalizzazione rigorosa tra 0 e 1
        sal_min, sal_max = sal_smooth.min(), sal_smooth.max()
        if sal_max > sal_min:
            sal_norm = (sal_smooth - sal_min) / (sal_max - sal_min)
        else:
            sal_norm = np.zeros_like(sal_smooth)
            
        # 3. Usiamo la colormap 'hot' o 'OrRd' (Arancione-Rosso) per l'overlay.
        # Questo evita il blu e il verde che sporcano l'immagine.
        cmap = plt.get_cmap('hot') 
        rgba_heatmap = cmap(sal_norm)
        
        # 4. SOGLIATURA AGGRESSIVA (Isoliamo solo i picchi di attivazione)
        # Invece di usare i percentili globali, prendiamo solo le zone 
        # che hanno un'intensità superiore al 50% del picco massimo dell'immagine.
        # Tutto il resto diventa completamente trasparente (Alpha = 0).
        threshold = 0.5  
        low_importance_mask = sal_norm < threshold
        rgba_heatmap[low_importance_mask, 3] = 0.0  # Rende invisibile lo sfondo e le aree neutre
        
        # Per le aree importanti, moduliamo l'opacità per non coprire troppo l'osso
        high_importance_mask = ~low_importance_mask
        rgba_heatmap[high_importance_mask, 3] = sal_norm[high_importance_mask] * 0.7

        # --- PLOTTING ---
        
        # Colonna 1: Immagine originale pulita
        axes[i, 0].imshow(img, cmap='gray')
        if i == 0: axes[i, 0].set_title('Original', fontsize=15)
        axes[i, 0].axis('off')
        
        # Colonna 2: Mappa di Occlusion Pura (Manteniamo 'jet' o 'turbo' qui, 
        # perché nella mappa isolata serve vedere l'intero spettro)
        axes[i, 1].imshow(sal_smooth, cmap='jet')
        if i == 0: axes[i, 1].set_title(title, fontsize=15)
        axes[i, 1].axis('off')
        
        # Colonna 3: Overlay Chirurgico (Radiografia + Solo Picchi di Calore)
        axes[i, 2].imshow(img, cmap='gray')
        axes[i, 2].imshow(rgba_heatmap)  # Mostra solo i focolai caldi sopra l'osso
        if i == 0: axes[i, 2].set_title('Occlusion Overlay (Clean)', fontsize=15)
        axes[i, 2].axis('off')

    plt.tight_layout()


    
def _compute_overlay_rgba(method_key, sal):
    '''
    Calcola l'overlay RGBA per un singolo metodo, replicando la logica
    delle funzioni visualize_triplet_* esistenti, in base al nome del metodo.
    Ritorna: (overlay_rgba oppure None, colormap_da_usare_se_no_overlay)
    '''
    key = method_key.lower()
    sal = np.asarray(sal)
    abs_sal = np.abs(sal)

    # --- Input x Gradient ---
    if 'gradient' in key or 'ixg' in key or 'input_x' in key:
        v_max = np.percentile(abs_sal, 99.5)
        if v_max == 0: v_max = 1
        sal_norm = np.clip(sal, -v_max, v_max) / v_max
        threshold = np.percentile(abs_sal, 80)
        overlay_rgba = np.zeros((*sal.shape, 4))
        pos_mask = (sal > 0) & (abs_sal > threshold)
        neg_mask = (sal < 0) & (abs_sal > threshold)
        overlay_rgba[pos_mask, 1] = 1.0
        overlay_rgba[pos_mask, 3] = np.abs(sal_norm[pos_mask]) * 0.85
        overlay_rgba[neg_mask, 0] = 1.0
        overlay_rgba[neg_mask, 3] = np.abs(sal_norm[neg_mask]) * 0.85
        return overlay_rgba

    # --- Integrated Gradients ---
    if key == 'ig' or 'integrated' in key:
        v_max = np.percentile(abs_sal, 99)
        if v_max == 0: v_max = 1
        sal_norm = np.clip(sal, -v_max, v_max) / v_max
        threshold = np.percentile(abs_sal, 85)
        overlay_rgba = np.zeros((*sal.shape, 4))
        pos_mask = (sal > 0) & (abs_sal > threshold)
        neg_mask = (sal < 0) & (abs_sal > threshold)
        overlay_rgba[pos_mask, 1] = 1.0
        overlay_rgba[pos_mask, 3] = np.abs(sal_norm[pos_mask]) * 0.8
        overlay_rgba[neg_mask, 0] = 1.0
        overlay_rgba[neg_mask, 3] = np.abs(sal_norm[neg_mask]) * 0.8
        return overlay_rgba

    # --- LIME ---
    if 'lime' in key:
        max_val = abs_sal.max() if abs_sal.max() > 0 else 1
        threshold = np.percentile(abs_sal, 75)
        overlay_rgba = np.zeros((*sal.shape, 4))
        pos_mask = (sal > 0) & (abs_sal > threshold)
        neg_mask = (sal < 0) & (abs_sal > threshold)
        overlay_rgba[pos_mask, 1] = 1.0
        overlay_rgba[pos_mask, 3] = (abs_sal[pos_mask] / max_val) * 0.6
        overlay_rgba[neg_mask, 0] = 1.0
        overlay_rgba[neg_mask, 3] = (abs_sal[neg_mask] / max_val) * 0.6
        return overlay_rgba

    # --- SHAP ---
    if 'shap' in key:
        max_val = abs_sal.max() if abs_sal.max() > 0 else 1
        threshold = np.percentile(abs_sal, 75)
        overlay_rgba = np.zeros((*sal.shape, 4))
        pos_mask = (sal > 0) & (abs_sal > threshold)
        neg_mask = (sal < 0) & (abs_sal > threshold)
        overlay_rgba[pos_mask, 0] = 1.0
        overlay_rgba[pos_mask, 3] = (abs_sal[pos_mask] / max_val) * 0.6
        overlay_rgba[neg_mask, 2] = 1.0
        overlay_rgba[neg_mask, 3] = (abs_sal[neg_mask] / max_val) * 0.6
        return overlay_rgba

    # --- Occlusion ---
    if 'occlusion' in key:
        sal_smooth = gaussian_filter(sal, sigma=2.0)
        sal_min, sal_max = sal_smooth.min(), sal_smooth.max()
        if sal_max > sal_min:
            sal_norm = (sal_smooth - sal_min) / (sal_max - sal_min)
        else:
            sal_norm = np.zeros_like(sal_smooth)
        cmap = plt.get_cmap('hot')
        rgba_heatmap = cmap(sal_norm)
        threshold = 0.5
        low_mask = sal_norm < threshold
        rgba_heatmap[low_mask, 3] = 0.0
        high_mask = ~low_mask
        rgba_heatmap[high_mask, 3] = sal_norm[high_mask] * 0.7
        return rgba_heatmap

    # --- Fallback: saliency-style (hot cmap, thresholded alpha) ---
    sal_s = gaussian_filter(np.abs(sal), sigma=1.5)
    s_min, s_max = sal_s.min(), sal_s.max()
    sal_norm = (sal_s - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(sal_s)
    cmap = plt.get_cmap('hot')
    rgba_heatmap = cmap(sal_norm)
    vmin_thr = np.percentile(sal_norm, 80)
    low_mask = sal_norm < vmin_thr
    rgba_heatmap[low_mask, 3] = 0.0
    high_mask = ~low_mask
    rgba_heatmap[high_mask, 3] = 0.5
    return rgba_heatmap


def visualize_all_methods_row(idx, inputs, attribution_all, title, method_keys, figsize_per_col=4):
    '''
    Visualizza in un'unica riga: immagine originale + overlay di tutti i metodi
    indicati in method_keys, per un singolo indice del batch.

    Parameters
    ----------
    idx : int
        Indice dell'immagine in inputs da visualizzare
    inputs : batch di immagini
    attribution_all : dict {method_key: attribuzioni (batch, H, W)}
    title : titolo generale della figura
    method_keys : list[str]
        Lista di chiavi da cercare in attribution_all
    '''
    n_cols = len(method_keys) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(figsize_per_col * n_cols, figsize_per_col))

    img = inputs[idx].mean(0).cpu().numpy()

    # Colonna 0: immagine originale
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original', fontsize=13)
    axes[0].axis('off')

    # Colonne successive: overlay per ciascun metodo
    for col, method_key in enumerate(method_keys, start=1):
        method_attributions = attribution_all.get(method_key)
        if method_attributions is None:
            axes[col].axis('off')
            axes[col].set_title(f'{method_key} (N/A)', fontsize=11)
            continue

        sal = method_attributions[idx]
        # se è un tensore torch, portalo su cpu/numpy
        if hasattr(sal, 'cpu'):
            sal = sal.cpu().numpy()

        overlay_rgba = _compute_overlay_rgba(method_key, sal)

        axes[col].imshow(img, cmap='gray')
        axes[col].imshow(overlay_rgba)
        axes[col].set_title(method_key, fontsize=11)
        axes[col].axis('off')

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    return fig