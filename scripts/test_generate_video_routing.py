import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_video as video


class LoraExpertRoutingTests(unittest.TestCase):
    def setUp(self):
        self.original_dual_expert = video.DUAL_EXPERT
        video.DUAL_EXPERT = True

    def tearDown(self):
        video.DUAL_EXPERT = self.original_dual_expert

    def test_unscoped_lora_uses_high_noise_expert_only(self):
        path = "/models/JSWANMaid-000009.safetensors"
        self.assertEqual(video._expert_lora_path(False, path), path)
        self.assertIsNone(video._expert_lora_path(True, path))

    def test_explicit_low_lora_uses_low_noise_expert_only(self):
        path = "/models/PussyLoRA_wan2.2low_epoch80.safetensors"
        self.assertIsNone(video._expert_lora_path(False, path))
        self.assertEqual(video._expert_lora_path(True, path), path)

    def test_explicit_high_lora_uses_high_noise_expert_only(self):
        path = "/models/character_A14B_HIGH_epoch10.safetensors"
        self.assertEqual(video._expert_lora_path(False, path), path)
        self.assertIsNone(video._expert_lora_path(True, path))

    def test_highres_name_is_not_mistaken_for_high_noise_marker(self):
        path = "/models/highresfix-v1.safetensors"
        self.assertEqual(video._lora_noise_marker(path), None)
        self.assertIsNone(video._expert_lora_path(True, path))

    def test_existing_high_low_pair_routes_one_file_per_expert(self):
        with tempfile.TemporaryDirectory() as directory:
            high = Path(directory, "Wan22-Lightning-T2V-A14B-4step-HIGH.safetensors")
            low = Path(directory, "Wan22-Lightning-T2V-A14B-4step-LOW.safetensors")
            high.touch()
            low.touch()

            self.assertEqual(video._expert_lora_path(False, str(high)), str(high))
            self.assertEqual(video._expert_lora_path(True, str(high)), str(low))
            self.assertEqual(video._expert_lora_path(False, str(low)), str(high))
            self.assertEqual(video._expert_lora_path(True, str(low)), str(low))

    def test_single_denoiser_keeps_selected_lora(self):
        video.DUAL_EXPERT = False
        path = "/models/explicit_LOW.safetensors"
        self.assertEqual(video._expert_lora_path(False, path), path)
        self.assertEqual(video._expert_lora_path(True, path), path)

    def test_dual_expert_uses_reference_low_noise_guidance(self):
        self.assertEqual(video._low_noise_guidance({}, 4.0), 3.0)
        self.assertEqual(video._low_noise_guidance({}, 1.0), 1.0)
        self.assertEqual(video._low_noise_guidance({"cfg_scale_2": 2.5}, 4.0), 2.5)

    def test_single_denoiser_keeps_one_guidance_value(self):
        video.DUAL_EXPERT = False
        self.assertEqual(video._low_noise_guidance({}, 4.0), 4.0)

    def test_short_standard_vae_clip_uses_full_frame_decode(self):
        self.assertTrue(video._use_untiled_vae_decode(16, 5, 60, 104))

    def test_long_or_heavy_vae_clip_keeps_tiled_decode(self):
        self.assertFalse(video._use_untiled_vae_decode(16, 13, 60, 104))
        self.assertFalse(video._use_untiled_vae_decode(48, 5, 60, 104))


if __name__ == "__main__":
    unittest.main()
