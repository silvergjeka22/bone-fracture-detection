import os
import torch


def save_attributions(attributions_dict):
    """
    Save a dictionary containing attribution maps.

    Parameters
    ----------
    attributions_dict : dict
        Dictionary:
            key -> list of attribution maps
    ----------
    """

    for key in attributions_dict.keys():
        torch.save(attributions_dict.get(key), key)


def load_attributions():
    '''
        From actual foleder load the saved attribution map.

        Return a dict{xai_method_name : attribution}.
        
    '''

    attributions = {}

    for filename in os.listdir():

        # prende solo i file senza estensione
        if os.path.splitext(filename)[1] == "":
            
            attributions[filename] = torch.load(filename,
                map_location="cpu",
                weights_only=False
            )

    return attributions