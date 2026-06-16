import torch

ckpt = torch.load('D:/EEG-FM-Bench/dashboard/models/eegpt_unified_epoch_6.pt',
                  map_location='cpu', weights_only=False)
state = ckpt['model_state_dict']

print("=== Сите classifier keys ===")
for k, v in state.items():
    if 'classifier' in k.lower() or 'head' in k.lower():
        print(f"  {k}  →  shape {tuple(v.shape)}")

print("\n=== Encoder keys (првите 20) ===")
encoder_keys = [k for k in state.keys() if 'encoder' in k.lower()]
for k in encoder_keys[:20]:
    print(f"  {k}  →  shape {tuple(state[k].shape)}")
print(f"  ...total {len(encoder_keys)} encoder keys")