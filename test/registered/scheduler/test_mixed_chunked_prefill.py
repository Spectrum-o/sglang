import math
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
import zmq

from sglang.srt.environ import envs
from sglang.srt.observability.forward_pass_metrics import decode
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=184, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd")


class TestMixedChunkedPrefill(GSM8KMixin, CustomTestCase):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_accuracy_thres = 0.62

    extra_args = [
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "32",
    ]

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.fpm_endpoint = f"ipc://{directory.name}/fpm"
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=cls.extra_args
                + [
                    "--return-output-ids",
                    "--enable-forward-pass-metrics",
                    "--forward-pass-metrics-ipc-name",
                    cls.fpm_endpoint,
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_logprobs(self):
        # Distinct token IDs expose row/position mixups across chunk boundaries.
        prompts = [[100], list(range(101, 230)), list(range(1001, 1130))]
        deadline = time.monotonic() + 60

        def generate(index):
            response = requests.post(
                self.base_url + "/generate",
                json=dict(
                    input_ids=prompts[index],
                    sampling_params=dict(
                        temperature=0,
                        max_new_tokens=128 if index == 0 else 8,
                        ignore_eos=True,
                    ),
                    return_logprob=index != 1,
                    logprob_start_len=0,
                    top_logprobs_num=5 if index != 1 else 0,
                ),
                timeout=max(1, deadline - time.monotonic()),
            )
            response.raise_for_status()
            return response.json()

        with (
            zmq.Context() as context,
            context.socket(zmq.SUB) as subscriber,
            ThreadPoolExecutor(max_workers=3) as pool,
        ):
            subscriber.setsockopt(zmq.SUBSCRIBE, b"")
            subscriber.connect(self.fpm_endpoint + ".0")
            futures, mixed = [], 0
            while time.monotonic() < deadline:
                if subscriber.poll(100):
                    frame = decode(subscriber.recv_multipart()[-1])
                    metrics = frame.scheduled_requests
                    is_mixed = bool(
                        metrics.num_prefill_requests and metrics.num_decode_requests
                    )
                    mixed += is_mixed
                    # Wait for subscription, then decode, then plain-prefill + LP-decode.
                    if (
                        (not futures and frame.wall_time == 0)
                        or (len(futures) == 1 and metrics.num_decode_requests)
                        or (len(futures) == 2 and is_mixed)
                    ):
                        futures.append(pool.submit(generate, len(futures)))
                if len(futures) == 3 and all(f.done() for f in futures):
                    break
            else:
                self.fail("Timed out waiting for requests and MIXED telemetry")
        results = [f.result() for f in futures]
        self.assertGreater(mixed, 0, "Logprob requests never executed MIXED")
        print(f"{type(self).__name__}: {mixed} MIXED forwards", flush=True)
        for index, (ids, response) in enumerate(zip(prompts, results)):
            meta = response["meta_info"]
            self.assertEqual(len(response["output_ids"]), 128 if index == 0 else 8)
            if index == 1:
                continue
            for side, tokens in (
                ("input", ids),
                ("output", response["output_ids"]),
            ):
                rows, top = (
                    meta[f"{side}_token_logprobs"],
                    meta[f"{side}_top_logprobs"],
                )
                self.assertEqual([row[1] for row in rows], tokens)
                self.assertEqual(len(top), len(tokens))
                for i, ((value, token, *_), candidates) in enumerate(zip(rows, top)):
                    if side == "input" and i == 0:
                        self.assertIsNone(value)
                        continue
                    self.assertTrue(math.isfinite(value) and value <= 1e-6)
                    scores = {t: v for v, t, *_ in candidates}
                    self.assertEqual((len(candidates), len(scores)), (5, 5))
                    self.assertTrue(all(math.isfinite(v) for v in scores.values()))
                    self.assertEqual(
                        list(scores.values()),
                        sorted(scores.values(), reverse=True),
                    )
                    if token in scores:
                        self.assertAlmostEqual(value, scores[token], delta=3e-6)


class TestMixedChunkedPrefillNoRadixCache(TestMixedChunkedPrefill):
    extra_args = [
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "32",
        "--disable-radix-cache",
    ]


if __name__ == "__main__":
    unittest.main()
