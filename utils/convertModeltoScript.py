import torch

from utils.teambuild_model import TeamBuilderModel

"""
Notes:
TorchScript can not be used in multiprocessing. Use normal model state dict for submission.
"""

LOAD_PATH = r""
SAVE_PATH = r""


# Build Model
model = TeamBuilderModel()

# Load Model State
model.load_state_dict(torch.load(LOAD_PATH, weights_only=True))

# Set it to eval mode,
# this should apprently be done to prevent issues when saving?
model.eval()

# Save Model as Torchscript
# Script the model, better than tracing for models with control flow
scripted_model = torch.jit.script(model)

# Save to file
scripted_model.save(SAVE_PATH)