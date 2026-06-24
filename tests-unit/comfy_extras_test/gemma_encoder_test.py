import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib
import torch

# Import custom node with hyphen in its name
gemma_encoder = importlib.import_module("custom_nodes.ComfyUI-LTXVideo.gemma_encoder")
find_matching_dir = gemma_encoder.find_matching_dir
LTXVGemmaCLIPModelLoader = gemma_encoder.LTXVGemmaCLIPModelLoader


def test_find_matching_dir(tmp_path):
    # Create some nested dummy files
    sub = tmp_path / "subdir"
    sub.mkdir()
    target_file = sub / "test_pattern.txt"
    target_file.touch()
    
    # Matching pattern
    matched = find_matching_dir(str(tmp_path), "test_pattern.txt")
    assert Path(matched) == sub
    
    # Missing pattern
    with pytest.raises(FileNotFoundError):
        find_matching_dir(str(tmp_path), "non_existent.txt")


@patch.object(gemma_encoder.folder_paths, "get_full_path")
@patch.object(gemma_encoder, "AutoImageProcessor")
@patch.object(gemma_encoder, "Gemma3Processor")
@patch.object(gemma_encoder, "ltxv_gemma_clip")
@patch.object(gemma_encoder.comfy.sd, "CLIP")
def test_gemma_loader_fallback(mock_clip, mock_ltxv_clip, mock_gemma_processor, mock_from_pretrained, mock_get_full_path):
    # Setup paths
    fake_safetensors = "/fake/models/text_encoders/gemma_model.safetensors"
    mock_get_full_path.side_effect = lambda type_name, path: fake_safetensors if type_name == "text_encoders" else f"/fake/models/checkpoints/{path}"
    
    # Instantiate loader
    loader = LTXVGemmaCLIPModelLoader()
    
    # We expect this to run and call ltxv_gemma_clip with gemma_model_path pointing to our fake file,
    # and falling back tokenizer/processor paths to the bundled gemma_configs directory
    loader.load_model(gemma_path="gemma_model.safetensors", ltxv_path="ltxv.safetensors", max_length=1024)
    
    # Verify the mocked calls:
    # 1. ltxv_gemma_clip should be called with the path to the single safetensors file as the first arg
    mock_ltxv_clip.assert_called_once()
    args, kwargs = mock_ltxv_clip.call_args
    assert args[0] == Path(fake_safetensors)
    
    # 2. AutoImageProcessor.from_pretrained should be called with the local gemma_configs directory path
    mock_from_pretrained.from_pretrained.assert_called_once()
    config_dir_arg = mock_from_pretrained.from_pretrained.call_args[0][0]
    assert "gemma_configs" in config_dir_arg


def test_fork_rng_for_device_cpu():
    device = torch.device("cpu")
    with gemma_encoder.fork_rng_for_device(device):
        torch.manual_seed(42)
        r1 = torch.rand(5)
    # Outside context, seed changes
    torch.manual_seed(100)
    # Re-entering should restore state
    with gemma_encoder.fork_rng_for_device(device):
        torch.manual_seed(42)
        r2 = torch.rand(5)
    assert torch.equal(r1, r2)


def test_fork_rng_for_device_mps(monkeypatch):
    device = MagicMock()
    device.type = "mps"
    
    mock_get_rng_state = MagicMock(return_value=b"fake_state")
    mock_set_rng_state = MagicMock()
    
    # We patch torch.mps to test the mps branch safely on non-MPS systems too
    with patch("torch.mps.get_rng_state", mock_get_rng_state), \
         patch("torch.mps.set_rng_state", mock_set_rng_state), \
         patch("torch.mps.manual_seed", MagicMock()):
        with gemma_encoder.fork_rng_for_device(device):
            pass
            
    mock_get_rng_state.assert_called_once()
    mock_set_rng_state.assert_called_once_with(b"fake_state")
