import json
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_sdxl as sdxl


def _write_model_index(dir_path, class_name):
    Path(dir_path, "model_index.json").write_text(json.dumps({"_class_name": class_name}))


def _write_scheduler_config(dir_path, **fields):
    sched_dir = Path(dir_path, "scheduler")
    sched_dir.mkdir(exist_ok=True)
    Path(sched_dir, "scheduler_config.json").write_text(json.dumps(fields))


class ArchitectureOfTests(unittest.TestCase):
    def test_sd3_detected_before_generic_stablediffusion_substring(self):
        with tempfile.TemporaryDirectory() as d:
            _write_model_index(d, "StableDiffusion3Pipeline")
            self.assertEqual(sdxl.architecture_of(d, "StableDiffusion3Pipeline")["family"], "sd3")

    def test_plain_sdxl_with_no_scheduler_config_is_not_turbo(self):
        with tempfile.TemporaryDirectory() as d:
            _write_model_index(d, "StableDiffusionXLPipeline")
            arch = sdxl.architecture_of(d, "StableDiffusionXLPipeline")
            self.assertEqual(arch["family"], "sdxl")
            self.assertTrue(arch["clamp_resolution"] is False)

    def test_sdxl_turbo_scheduler_signature_detected(self):
        with tempfile.TemporaryDirectory() as d:
            _write_model_index(d, "StableDiffusionXLPipeline")
            _write_scheduler_config(d, timestep_spacing="trailing", interpolation_type="linear")
            arch = sdxl.architecture_of(d, "StableDiffusionXLPipeline")
            self.assertEqual(arch["family"], "sdxl_turbo")
            self.assertEqual(arch["default_steps"], 4)
            self.assertLess(arch["default_cfg"], 2.0)

    def test_plain_sd15_with_epsilon_prediction_stays_sd15(self):
        with tempfile.TemporaryDirectory() as d:
            _write_model_index(d, "StableDiffusionPipeline")
            _write_scheduler_config(d, prediction_type="epsilon")
            self.assertEqual(sdxl.architecture_of(d, "StableDiffusionPipeline")["family"], "sd15")

    def test_v_prediction_scheduler_signature_detected_as_sd2(self):
        with tempfile.TemporaryDirectory() as d:
            _write_model_index(d, "StableDiffusionPipeline")
            _write_scheduler_config(d, prediction_type="v_prediction")
            arch = sdxl.architecture_of(d, "StableDiffusionPipeline")
            self.assertEqual(arch["family"], "sd2")
            self.assertEqual(arch["max_resolution"], 768)

    def test_sd_turbo_scheduler_signature_detected(self):
        with tempfile.TemporaryDirectory() as d:
            _write_model_index(d, "StableDiffusionPipeline")
            _write_scheduler_config(d, timestep_spacing="trailing", interpolation_type="linear")
            self.assertEqual(sdxl.architecture_of(d, "StableDiffusionPipeline")["family"], "sd_turbo")

    def test_unrecognized_class_name_falls_back_to_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            _write_model_index(d, "WanPipeline")
            self.assertEqual(sdxl.architecture_of(d, "WanPipeline")["family"], "unknown")

    def test_checkpoint_descriptor_matches_is_xl_flag(self):
        self.assertEqual(sdxl.architecture_of_checkpoint(True)["family"], "sdxl")
        self.assertEqual(sdxl.architecture_of_checkpoint(False)["family"], "sd15")

    def test_every_descriptor_has_the_same_field_set(self):
        keys = set(sdxl._DESCRIPTORS["sd15"].keys())
        for family, desc in sdxl._DESCRIPTORS.items():
            self.assertEqual(set(desc.keys()), keys, f"{family} descriptor field set drifted")


class _FakeTok:
    """Minimal stand-in for a HF tokenizer — returns one 'token' per word so tests
    don't need to load real CLIP/T5 weights to check which tokenizer gets used."""
    def __call__(self, text, truncation=False):
        class _Out:
            input_ids = list(range(max(1, len(text.split()))))
        return _Out()


class TokenLimitRoutingTests(unittest.TestCase):
    def test_clip_only_pipe_measures_against_tokenizer_and_tokenizer_2(self):
        class FakePipe:
            tokenizer = _FakeTok()
            tokenizer_2 = _FakeTok()
            tokenizer_3 = None
        # Compaction drops whole comma-clauses from the end — matches how every prompt
        # in this app is actually built (asset-guard/negative terms are comma lists).
        text = ", ".join(f"clause {i} has several words in it" for i in range(20))
        out, count, trimmed = sdxl._fit_to_token_limit(FakePipe(), text, limit=77, use_t5=False)
        self.assertTrue(trimmed)
        self.assertLessEqual(count, 77)

    def test_sd3_use_t5_measures_against_tokenizer_3_not_clip(self):
        class FakePipe:
            tokenizer = _FakeTok()
            tokenizer_2 = _FakeTok()
            tokenizer_3 = _FakeTok()
        # 100 words would blow CLIP's 77 limit but comfortably fits T5's 256.
        text = " ".join(["word"] * 100)
        out, count, trimmed = sdxl._fit_to_token_limit(FakePipe(), text, limit=256, use_t5=True)
        self.assertFalse(trimmed)
        self.assertEqual(out, text)


class ValidateOutputImageTests(unittest.TestCase):
    def test_normal_random_image_passes(self):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        sdxl._validate_output_image(Image.fromarray(arr), "sd15")  # must not raise

    def test_solid_black_image_is_rejected(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            sdxl._validate_output_image(Image.fromarray(arr), "sd3")

    def test_solid_color_image_is_rejected(self):
        arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            sdxl._validate_output_image(Image.fromarray(arr), "sdxl")


class GenerateImageInputValidationTests(unittest.TestCase):
    def test_zero_or_negative_or_misaligned_resolution_is_rejected_loudly(self):
        class FakePipe:
            _saient_arch = dict(sdxl._DESCRIPTORS["sd15"])
        for bad_w, bad_h in [(0, 512), (512, 0), (-8, 512), (513, 512), (512, 511)]:
            with self.assertRaises(RuntimeError):
                sdxl.generate_image(FakePipe(), "cpu", {"width": bad_w, "height": bad_h})

    def test_zero_or_negative_steps_is_rejected_loudly(self):
        class FakePipe:
            _saient_arch = dict(sdxl._DESCRIPTORS["sd15"])
        for bad_steps in (0, -1, -20):
            with self.assertRaises(RuntimeError):
                sdxl.generate_image(FakePipe(), "cpu", {"steps": bad_steps})


class DetailFacesCapabilityTests(unittest.TestCase):
    def test_unsupported_architecture_skips_without_touching_cv2(self):
        class FakePipe:
            _saient_arch = dict(sdxl._DESCRIPTORS["sd3"])
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        image, n_fixed, reason = sdxl._detail_faces(FakePipe(), "cpu", {}, img)
        self.assertEqual(reason, "unsupported")
        self.assertEqual(n_fixed, 0)
        self.assertIs(image, img)


if __name__ == "__main__":
    unittest.main()
