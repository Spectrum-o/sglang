import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
import zmq
from transformers import AutoTokenizer

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
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        prompts = [
            tokenizer.encode(f"Request {i}: " + "I have a very good idea on this. " * i)
            for i in range(32)
        ]

        def generate(ids):
            response = requests.post(
                self.base_url + "/generate",
                json=dict(
                    input_ids=ids,
                    sampling_params=dict(
                        temperature=0, max_new_tokens=32, ignore_eos=True
                    ),
                    return_logprob=True,
                    logprob_start_len=0,
                    top_logprobs_num=5,
                ),
                timeout=600,
            )
            response.raise_for_status()
            return response.json()

        # Existing scheduler telemetry distinguishes a real MIXED forward from
        # separate prefill/decode passes; no model or scheduler instrumentation.
        with zmq.Context() as context, context.socket(zmq.SUB) as subscriber:
            subscriber.setsockopt(zmq.SUBSCRIBE, b"")
            subscriber.connect(self.fpm_endpoint + ".0")
            self.assertTrue(subscriber.poll(10000), "No forward-pass telemetry")
            while subscriber.poll(0):
                subscriber.recv_multipart()
            with ThreadPoolExecutor(max_workers=16) as pool:
                results = list(pool.map(generate, prompts))
            mixed = 0
            while subscriber.poll(200):
                metrics = decode(subscriber.recv_multipart()[-1]).scheduled_requests
                mixed += (
                    metrics.num_prefill_requests > 0 and metrics.num_decode_requests > 0
                )
            self.assertGreater(mixed, 0, "Logprob requests never executed MIXED")
            print(f"{type(self).__name__}: {mixed} MIXED forwards", flush=True)
        for ids, response in zip(prompts, results):
            meta = response["meta_info"]
            self.assertEqual(meta["completion_tokens"], 32)
            self.assertEqual(len(response["output_ids"]), 32)
            for side, tokens in (("input", ids), ("output", response["output_ids"])):
                rows, top = meta[f"{side}_token_logprobs"], meta[f"{side}_top_logprobs"]
                self.assertEqual([row[1] for row in rows], tokens)
                self.assertEqual(len(top), len(tokens))
                for i, ((value, token, *_), candidates) in enumerate(zip(rows, top)):
                    if side == "input" and i == 0:
                        self.assertIsNone(value)
                        continue
                    self.assertTrue(math.isfinite(value) and value <= 1e-6)
                    self.assertEqual(len(candidates), 5)
                    scores = {t: v for v, t, *_ in candidates}
                    self.assertEqual(len(scores), 5)
                    self.assertTrue(all(math.isfinite(v) for v in scores.values()))
                    self.assertEqual(
                        list(scores.values()), sorted(scores.values(), reverse=True)
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
