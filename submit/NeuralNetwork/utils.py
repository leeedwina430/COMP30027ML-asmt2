#######################################
#  code for save and load the models  #
#######################################

import torch

# Save and Load Functions
def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    # print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location=device)
    # print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return