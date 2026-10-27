import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mars_image_classification.data import class_mapping, load_manifest, preprocess_image
from mars_image_classification.metrics import classification_metrics
from mars_image_classification.workflows import prepare

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


class CoreTests(unittest.TestCase):
    def test_mapping_is_stable_and_excludes_other_by_default(self):
        mapping = class_mapping()
        self.assertEqual(mapping["crater"], 0)
        self.assertEqual(mapping["spider"], 6)
        self.assertNotIn("other", mapping)
        self.assertEqual(class_mapping(True)["other"], 0)

    def test_preprocessing_has_efficientnet_shape_and_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            Image.fromarray(np.full((11, 17), 127, dtype=np.uint8)).save(path)
            image = preprocess_image(path, 32)
        self.assertEqual(image.shape, (32, 32, 3))
        self.assertEqual(image.dtype, np.float32)
        np.testing.assert_array_equal(image[..., 0], image[..., 2])

    def test_manifest_resolves_official_split_and_filters_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "map-proj-v3_2"
            images.mkdir()
            for filename in ("a.png", "b.png"):
                Image.new("L", (4, 4)).save(images / filename)
            (root / "labels-map-proj-v3_2_train_val_test.txt").write_text(
                "a.png 1 train\nb.png 0 test\n", encoding="utf-8"
            )
            rows = load_manifest(root)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].source_label, rows[0].split), (1, "train"))

    def test_prepare_reports_split_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "map-proj-v3_2"
            images.mkdir()
            rows = []
            for index, split in enumerate(("train", "val", "test")):
                filename = f"{index}.png"
                Image.new("L", (4, 4)).save(images / filename)
                rows.append(f"{filename} {index + 1} {split}")
            (root / "labels-map-proj-v3_2_train_val_test.txt").write_text(
                "\n".join(rows), encoding="utf-8"
            )
            report = prepare(str(root))
        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["splits"], {"train": 1, "val": 1, "test": 1})

    def test_metrics_known_predictions(self):
        result = classification_metrics([0, 1, 1, 2], [0, 1, 0, 2], 3)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 2 / 3 + 1) / 3)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [1, 1, 0], [0, 0, 1]])

    @unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow optional dependency is not installed")
    def test_model_checkpoint_reload_and_prediction_structure(self):
        from mars_image_classification.modeling import build_model
        from mars_image_classification.workflows import predict

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.keras"
            model = build_model(7, image_size=32, weights=None)
            self.assertEqual(model.output_shape[-1], 7)
            model.save(checkpoint)
            (root / "metadata.json").write_text(json.dumps({
                "class_to_index": class_mapping(), "image_size": 32
            }), encoding="utf-8")
            image_path = root / "image.png"
            Image.new("L", (20, 20), 128).save(image_path)
            predictions = predict(str(checkpoint), str(image_path))
        self.assertEqual(len(predictions), 1)
        self.assertIn("label", predictions[0])
        self.assertIn("scores", predictions[0])
        self.assertEqual(len(predictions[0]["scores"]), 7)


if __name__ == "__main__":
    unittest.main()
