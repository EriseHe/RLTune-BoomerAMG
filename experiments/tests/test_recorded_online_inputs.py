"""Exact accepted inputs remain portable without weakening their hash guards."""

import copy
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.paper_final.common import online_inputs
from experiments.paper_final.online import case_loop
from experiments.paper_final.online.configuration import (
    parse_joint_experiment_config,
    runtime_config_from_spec,
)


ROOT = Path(__file__).resolve().parents[2]


class RecordedOnlineInputTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(
            (
                ROOT
                / "experiments/paper_final/04_online/20260920_formal/diffusion_40.json"
            ).read_text()
        )
        self.args = runtime_config_from_spec(parse_joint_experiment_config(self.raw))

    def test_accepted_stream_avoids_platform_math_and_retains_old_manifest(self):
        with patch.object(
            case_loop,
            "generate_problem_instances",
            side_effect=AssertionError("resampled"),
        ):
            stream, manifest = case_loop.build_paired_instance_stream(self.args)
        self.assertEqual(len(stream), 5000)
        expected = self.raw["stream"]["expected_sha256"]
        self.assertEqual(case_loop._instance_stream_hash(stream), expected)
        self.assertEqual(
            manifest,
            json.loads((online_inputs.STREAM_DIRECTORY / "manifest.json").read_text())[
                "streams"
            ][expected]["stream_manifest"],
        )
        self.assertEqual(stream[0][1].shape, (8,))
        self.assertEqual(stream[0][1].dtype, np.dtype(float))

    def test_every_sampling_setting_is_bound_to_the_accepted_hash(self):
        changes = {
            "problem": "scalar_anisotropic_diffusion_advection",
            "grid_shape": (41, 40, 40),
            "c_min": 2.0,
            "c_max": 999.0,
            "seed": self.args.seed + 1,
            "train_seed_groups": str(self.args.train_seed_groups) + ",99",
            "train_shuffle_seeds": str(int(self.args.train_shuffle_seeds) + 1),
            "instance_offset": 1,
            "train_cases": 4999,
            "train_cases_per_seed": 624,
            "train_group_take": 4999,
        }
        for name, value in changes.items():
            with self.subTest(setting=name):
                args = replace(self.args, **{name: value})
                with self.assertRaisesRegex(ValueError, "different sampling settings"):
                    case_loop.build_paired_instance_stream(args)
        raw = copy.deepcopy(self.raw)
        raw["problem"]["kind"] = "scalar_anisotropic_diffusion_advection"
        raw["problem"]["advection_min"] = 1.0
        raw["problem"]["advection_max"] = 1000.0
        # Changing PDE family also invalidates a recorded diffusion hash.
        args = runtime_config_from_spec(parse_joint_experiment_config(raw))
        with self.assertRaisesRegex(ValueError, "different sampling settings"):
            case_loop.build_paired_instance_stream(args)
        advection_raw = json.loads(
            (
                ROOT
                / "experiments/paper_final/04_online/20260920_formal/advection_40.json"
            ).read_text()
        )
        advection_args = runtime_config_from_spec(
            parse_joint_experiment_config(advection_raw)
        )
        for name, value in (("advection_min", 2.0), ("advection_max", 999.0)):
            with self.subTest(setting=name):
                args = replace(advection_args, **{name: value})
                with self.assertRaisesRegex(ValueError, "different sampling settings"):
                    case_loop.build_paired_instance_stream(args)

    def test_blob_and_canonical_content_have_independent_hash_checks(self):
        expected = self.raw["stream"]["expected_sha256"]
        original = json.loads(
            (online_inputs.STREAM_DIRECTORY / "manifest.json").read_text()
        )
        entry = original["streams"][expected]
        compressed = (online_inputs.STREAM_DIRECTORY / entry["file"]).read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest_path = directory / "manifest.json"
            manifest_path.write_text(json.dumps(original))
            blob_path = directory / entry["file"]
            blob_path.write_bytes(compressed + b"corrupt")
            with patch.object(online_inputs, "STREAM_DIRECTORY", directory):
                with self.assertRaisesRegex(
                    ValueError, "compressed file hash mismatch"
                ):
                    case_loop.build_paired_instance_stream(self.args)
                rows = gzip.decompress(compressed).splitlines()
                first = json.loads(rows[0])
                first["context"][1] += 0.01
                rows[0] = json.dumps(
                    first, sort_keys=True, separators=(",", ":")
                ).encode()
                changed = gzip.compress(b"\n".join(rows) + b"\n", mtime=0)
                blob_path.write_bytes(changed)
                original["streams"][expected]["compressed_sha256"] = hashlib.sha256(
                    changed
                ).hexdigest()
                manifest_path.write_text(json.dumps(original))
                with self.assertRaisesRegex(
                    ValueError, "canonical input hash mismatch"
                ):
                    case_loop.build_paired_instance_stream(self.args)

    def test_unpinned_and_unknown_streams_use_the_original_sampler(self):
        for expected in ("", "0" * 64):
            with self.subTest(expected=expected):
                args = replace(self.args, expected_stream_hash=expected)
                with patch.object(
                    case_loop,
                    "generate_problem_instances",
                    side_effect=RuntimeError("original sampler"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "original sampler"):
                        case_loop.build_paired_instance_stream(args)


if __name__ == "__main__":
    unittest.main()
