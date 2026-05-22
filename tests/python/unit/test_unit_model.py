from model import MusicModel
import pytest
import torch

def test_model_forward():
  model = MusicModel()

  batch_size = 16
  num_frames = 1290
  mfcc_size = 13
  x = torch.rand((batch_size, num_frames, mfcc_size))
  num_classes_expected = 10

  model.eval()
  with torch.no_grad():
    y = model(x)
  assert(y.numel() == batch_size * num_classes_expected), "Invalid size of the model output."

def test_model_save(tmp_path):
  file_path = tmp_path / "file.pt"
  model = MusicModel()
  model.save(file_path)
  assert(file_path.exists()), "Model is not saved."


@pytest.fixture(params=[True, False])
def exist(request):
  return request.param

def test_model_load(tmp_path, exist):
  file_path = tmp_path / "file.pt"
  model_save = MusicModel()
  model_load = MusicModel()

  if exist:
    model_save.save(file_path)
    model_load.load(file_path)
    for k, v in model_save.state_dict().items():
      assert(torch.equal(v, model_load.state_dict()[k])), "Model is not loaded properly."
  else:
    with pytest.raises(FileNotFoundError, match=f"MusicModel.load: File: {file_path} does not exist."):
      model_load.load(file_path)
