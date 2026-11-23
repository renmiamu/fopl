from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    from .evaluate_multiple_choice import (
        BenchmarkSpec,
        append_checkpoint,
        build_messages,
        build_prompt,
        extract_choice_answer,
        merge_checkpoint,
        new_payload,
        normalize_arc,
        normalize_gpqa,
        normalize_hellaswag,
        normalize_mmlu,
        normalize_openbookqa,
        formatted_answer_fallback,
        run_fingerprint,
        load_or_create_payload,
        boxed_contents,
        message_response_text,
    )
except ImportError:
    from evaluate_multiple_choice import (  # type: ignore[no-redef]
        BenchmarkSpec,
        append_checkpoint,
        build_messages,
        build_prompt,
        extract_choice_answer,
        merge_checkpoint,
        new_payload,
        normalize_arc,
        normalize_gpqa,
        normalize_hellaswag,
        normalize_mmlu,
        normalize_openbookqa,
        formatted_answer_fallback,
        run_fingerprint,
        load_or_create_payload,
        boxed_contents,
        message_response_text,
    )


CHOICES = [
    {"label": "A", "text": "alpha"},
    {"label": "B", "text": "beta"},
    {"label": "C", "text": "gamma"},
    {"label": "D", "text": "delta"},
]


class ChoiceParserTests(unittest.TestCase):
    def test_analysis_tag_preserves_final_answer(self) -> None:
        self.assert_parse(r"<analysis>Guess \boxed{A}</analysis> Final: C", "C")

    def test_response_content_and_reasoning_formats(self) -> None:
        cases = [
            ({"content": "Answer: C", "reasoning_content": "Answer: B"}, "Answer: C"),
            ({"content": None, "reasoning_content": r"\boxed{B}"}, r"\boxed{B}"),
            ({"content": [{"type": "text", "text": "Answer: A"}]}, "Answer: A"),
            (SimpleNamespace(content="", model_extra={"reasoning_content": "Answer: D"}), "Answer: D"),
        ]
        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(message_response_text(message), expected)
        with self.assertRaisesRegex(RuntimeError, "no response text"):
            message_response_text({"content": ""})

    def assert_parse(self, response: str, expected: str, method: str | None = None) -> None:
        answer, actual_method = extract_choice_answer(response, CHOICES)
        self.assertEqual(answer, expected)
        if method is not None:
            self.assertEqual(actual_method, method)

    def test_explicit_answer_formats(self) -> None:
        cases = {
            "Answer: C": "C",
            "The final answer is option (B).": "B",
            "正确答案：D": "D",
            "答案是 A。": "A",
            "Therefore, choice C.": "C",
            "Some reasoning\n\n**Answer: B**": "B",
            "Some reasoning\n\nAnswer: **D**": "D",
            "Some reasoning\n\nAnswer: 3": "C",
        }
        for response, expected in cases.items():
            with self.subTest(response=response):
                self.assert_parse(response, expected)

    def test_boxed_and_think_formats(self) -> None:
        self.assert_parse(r"work... \boxed{\text{B}}", "B", "boxed")
        self.assert_parse(r"work... \boxed{\text{Option B}}", "B", "boxed")
        self.assert_parse(r"work... \boxed{Final answer: C}", "C", "boxed")
        self.assert_parse(r"first guess \boxed{A}; correction \boxed{D}", "D", "boxed")
        self.assert_parse(r"work... \fbox{\textbf{C}}", "C", "boxed")
        self.assert_parse(r"work... boxed{Ｄ}", "D", "unescaped_boxed")
        self.assert_parse(r"work... \boxed{\text{gamma}}", "C", "boxed_choice_text")
        self.assert_parse(r"work... \boxed{C. gamma}", "C", "boxed_label_and_text")
        self.assert_parse(r"work... \boxed{B", "B", "truncated_box")
        self.assert_parse(r"reasoning... answer\boxed{C}garbage", "C", "boxed")
        self.assert_parse(
            "<THINK>A and D are tempting, but C is supported.</THINK>\nAnswer: C",
            "C",
        )

    def test_structured_and_conclusion_formats(self) -> None:
        self.assert_parse('{"final_answer": "B"}', "B", "structured_json")
        self.assert_parse('```json\n{"choice": 4}\n```', "D", "structured_json")
        self.assert_parse("After checking, I choose option C.", "C", "selection_suffix")
        self.assert_parse("综上所述，故选B。", "B", "selection_suffix")
        self.assert_parse("D is the correct answer.", "D", "selection_suffix")

    def test_conclusions_followed_by_degenerate_suffixes(self) -> None:
        cases = {
            "Solution complete. So answer B.التز": ("B", "explicit_conclusion_marker"),
            "finalized answer: Dإبد": ("D", "explicit_conclusion_marker"),
            "final_answer:B followed by noise": ("B", "explicit_conclusion_marker"),
            "The best answer is C._broken": ("C", "explicit_conclusion_marker"),
            "work\nanswer{A}garbage": ("A", "braced_answer_marker"),
            'ActionController.answer :final => "B"': ("B", "framework_answer_marker"),
            "Therefore, D is correct.إبد": ("D", "correctness_statement"),
            "因此选项C正确。garbage": ("C", "correctness_statement"),
            "After checking, so choose A. noise": ("A", "reasoning_selection"),
        }
        for response, (answer, method) in cases.items():
            with self.subTest(response=response):
                self.assert_parse(response, answer, method)

    def test_final_line_and_choice_text_fallbacks(self) -> None:
        self.assert_parse("Long explanation\n(C)", "C", "standalone_final_line")
        self.assert_parse("Long explanation\ngamma", "C", "exact_choice_text")
        self.assert_parse("Long explanation\nAnswer: gamma", "C", "explicit_choice_text")
        self.assert_parse("Long explanation ending in (D).", "D", "parenthesized_suffix")

    def test_does_not_treat_arbitrary_reasoning_label_as_answer(self) -> None:
        answer, method = extract_choice_answer(
            "Option A is implausible while option B has one advantage.", CHOICES
        )
        self.assertIsNone(answer)
        self.assertEqual(method, "not_found")
        answer, method = extract_choice_answer(
            "A package labelled boxed{B} appears, but option C still needs checking.",
            CHOICES,
        )
        self.assertIsNone(answer)
        self.assertEqual(method, "not_found")
        answer, method = extract_choice_answer(r"Final: \boxed{B or C}", CHOICES)
        self.assertIsNone(answer)
        self.assertEqual(method, "not_found")

    def test_rejects_out_of_range_label(self) -> None:
        answer, method = extract_choice_answer("Answer: E", CHOICES)
        self.assertIsNone(answer)
        self.assertEqual(method, "not_found")

    def test_formatted_answer_fallbacks(self) -> None:
        cases = {
            r"\textbf{Answer: A. alpha}garbage": "A",
            r"\textbf{Answer:} C. gamma.extra": "C",
            r"\textbf{Answer} C (gamma)garbage": "C",
            r"So, the answer is \(D\).noise": "D",
            r"The correct answer is \(\mathrm{A}\).noise": "A",
            "The answer to this question is C. gamma. noise": "C",
            "Thus, the answer should be B. noise": "B",
            "The correct answer should be option C. noise": "C",
            "The best answer is likely B. noise": "B",
            "The best answer from available choices is B. noise": "B",
            "finalized:C. gamma": "C",
            "Final: D": "D",
            "Final: *B*": "B",
            "Final: _C_": "C",
            "答案为C项。noise": "C",
            "最终答案应该是B。noise": "B",
            "Answer:\u200bB\u200d. noise": "B",
            r"work... \boxed{\textcolor{red}{B}}": "B",
            r"work... \boxed{\color{blue}C}": "C",
            r"work... \boxed{'D'}": "D",
            r"work... \boxed(B)": "B",
            r"work... \fbox[C]": "C",
            '<final_answer>"B"</final_answer>': "B",
            "<answer>gamma</answer>": "C",
            r"\begin{answer}" + "\nB. beta\nnoise": "B",
            "Reasoning...\n```json\n{\"selected_option\": \"D\"}\n```": "D",
            'Reasoning...\n{"final_option": "B"}': "B",
            "Reasoning...\n```text\nB\n```": "B",
            "Reasoning...\n```\ngamma\n```": "C",
            "So, after checking, I choose C. noise": "C",
            "Therefore, B is the best answer. noise": "B",
        }
        for response, answer in cases.items():
            with self.subTest(response=response):
                self.assert_parse(response, answer)

    def test_new_fallback_rejects_negation_hypotheses_and_ambiguity(self) -> None:
        cases = [
            "The answer should not be C.",
            r"\textbf{Answer: B or C}",
            r"\textbf{Answer: B/C}",
            r"\textbf{Answer: B and C}",
            "Final: B or C",
            "Final: A-D",
            "If the answer should be B, then we must check C.",
            "Suppose the answer would be C.",
            r"Print \textbf{Answer: B} as an example.",
            r"\boxed[B)",
            r"\boxed(B or C)",
            "<answer>B or C</answer>",
            "The answer should be Blue.",
            r"\textbf{Answer: E}",
            r"\textbf{Answer: 12}",
            r"\textbf{Answer:}" + "\nA. alpha\nB. beta\nC. gamma\nD. delta",
            'Consider this example: {"selected_option": "B"}\nStill checking.',
            "Another problem:\nA 36\nB 50\nC 72\nD 84\nE 120\nThe answer should be C.",
        ]
        for response in cases:
            with self.subTest(response=response):
                answer, method = extract_choice_answer(response, CHOICES)
                self.assertIsNone(answer)
                self.assertEqual(method, "not_found")

    def test_new_fallback_uses_last_explicit_conclusion(self) -> None:
        # Test fallback directly as legacy parser priorities are unchanged.
        self.assertEqual(formatted_answer_fallback(
            "Final: B. Correction: Final: D.", CHOICES, set("ABCD")
        )[0], "D")
        self.assertIsNone(formatted_answer_fallback(
            "Final: B. Correction: Final: E.", CHOICES, set("ABCD")
        )[0])
        self.assertIsNone(formatted_answer_fallback(
            "Final: B. Correction: Final: B or C.", CHOICES, set("ABCD")
        )[0])

    def test_deep_json_does_not_crash(self) -> None:
        answer, _ = extract_choice_answer('[' * 1500 + '0' + ']' * 1500, CHOICES)
        self.assertIsNone(answer)

    def test_nested_boxes_preserve_command_order(self) -> None:
        self.assertEqual(boxed_contents(r"\boxed{A}\boxed{\boxed{B}}"), [
            "A", r"\boxed{B}", "B",
        ])

    def test_runaway_unclosed_boxes_are_safe(self) -> None:
        malformed = r"\boxed{" * 10000
        self.assertEqual(boxed_contents(malformed), [])
        self.assert_parse(malformed + r"\boxed{C}", "C", "boxed")


class DatasetNormalizationTests(unittest.TestCase):
    def test_mmlu_integer_answer(self) -> None:
        row = normalize_mmlu(
            {
                "question": "q",
                "choices": ["one", "two", "three", "four"],
                "answer": 2,
                "subject": "college_physics",
            },
            0,
            42,
        )
        self.assertEqual(row["gold_answer"], "C")
        self.assertEqual(row["category"], "stem")

    def test_arc_numeric_source_labels(self) -> None:
        row = normalize_arc(
            {
                "id": "arc-1",
                "question": "q",
                "choices": {
                    "label": ["1", "2", "3", "4"],
                    "text": ["one", "two", "three", "four"],
                },
                "answerKey": "2",
            },
            0,
            42,
        )
        self.assertEqual(row["gold_answer"], "B")
        self.assertEqual([choice["label"] for choice in row["choices"]], list("ABCD"))

    def test_openbookqa_maps_source_answer_to_displayed_choice(self) -> None:
        item = {
            "id": "obqa-1",
            "question_stem": "Which material conducts electricity?",
            "choices": {
                "label": ["B", "A", "C", "D"],
                "text": ["wood", "copper", "glass", "rubber"],
            },
            "answerKey": "A",
            "fact1": "Copper is a conductive metal.",
        }
        row = normalize_openbookqa(item, 0, 42)
        self.assertEqual(row["problem_id"], "openbookqa/obqa-1/0")
        self.assertEqual(row["gold_answer"], "B")
        self.assertEqual(row["choices"][1]["text"], "copper")
        prompt = build_prompt(row)
        self.assertIn("B. copper", prompt)
        self.assertNotIn("Copper is a conductive metal.", prompt)

        with self.assertRaisesRegex(ValueError, "answerKey"):
            normalize_openbookqa({**item, "answerKey": "E"}, 0, 42)

    def test_hellaswag_zero_based_label(self) -> None:
        row = normalize_hellaswag(
            {
                "ind": 5,
                "ctx": "A person starts cooking",
                "endings": ["first", "second", "third", "fourth"],
                "label": "0",
                "activity_label": "Cooking",
            },
            0,
            42,
        )
        self.assertEqual(row["gold_answer"], "A")
        second = normalize_hellaswag(
            {
                "ind": 6,
                "ctx": "A person starts cooking",
                "endings": ["first", "second", "third", "fourth"],
                "label": "2",
                "activity_label": "Cooking",
            },
            1,
            42,
        )
        self.assertEqual(second["gold_answer"], "C")

    def test_gpqa_shuffle_is_deterministic_and_not_position_leaking(self) -> None:
        raw = {
            "Question": "A difficult question?",
            "Correct Answer": "correct",
            "Incorrect Answer 1": "wrong one",
            "Incorrect Answer 2": "wrong two",
            "Incorrect Answer 3": "wrong three",
            "High-level domain": "Physics",
        }
        first = normalize_gpqa(raw, 0, 42)
        second = normalize_gpqa(raw, 0, 42)
        self.assertEqual(first["choices"], second["choices"])
        self.assertEqual(first["gold_answer"], second["gold_answer"])
        selected = next(
            choice["text"]
            for choice in first["choices"]
            if choice["label"] == first["gold_answer"]
        )
        self.assertEqual(selected, "correct")


class PromptTests(unittest.TestCase):
    def test_prompt_requests_solve_and_box_without_including_gold_answer(self) -> None:
        target = {
            "problem_id": "mmlu/test",
            "question": "target",
            "choices": CHOICES,
            "gold_answer": "D",
        }
        prompt = build_prompt(target)
        self.assertEqual(
            prompt,
            "Question: target\n"
            "A. alpha\n"
            "B. beta\n"
            "C. gamma\n"
            "D. delta\n"
            r"Solve the problem. Put only the final option letter inside \boxed{}.",
        )
        self.assertNotIn(r"\boxed{D}", prompt)
        self.assertNotIn("step by step", prompt.casefold())

    def test_messages_are_zero_shot(self) -> None:
        prompt = "Question: target\nA. alpha\nB. beta"
        messages = build_messages(prompt, "Solve carefully.")
        self.assertEqual(messages, [
            {"role": "system", "content": "Solve carefully."},
            {"role": "user", "content": prompt},
        ])


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_recovers_once_and_regrades(self) -> None:
        spec = BenchmarkSpec(
            key="mmlu",
            display_name="MMLU",
            dataset="cais/mmlu",
            dataset_config="all",
            split="test",
            output=Path("unused.json"),
            system_prompt="system",
        )
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                model="model",
                base_url="http://localhost:8000/v1",
                dataset="cais/mmlu",
                dataset_config="all",
                split="test",
                max_tokens=100,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                seed=42,
                output=Path(directory) / "result.json",
                resume=True,
            )
            payload = new_payload(args, spec)
            legacy_identity = {
                "model": args.model,
                "base_url": args.base_url.rstrip("/"),
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "generation": payload["generation"],
                "benchmark": spec.key,
                "prompt_version": 2,
                "system_prompt": spec.system_prompt,
            }
            legacy_fingerprint = hashlib.sha256(json.dumps(
                legacy_identity, ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")).hexdigest()
            self.assertNotEqual(run_fingerprint(args, spec), legacy_fingerprint)
            self.assertEqual(payload["prompting"]["strategy"], "zero_shot_solve")
            legacy = {**payload, "prompt_version": 2, "run_fingerprint": legacy_fingerprint}
            args.output.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "run_fingerprint"):
                load_or_create_payload(args, spec)
            one_shot = {**payload, "prompt_version": 3, "run_fingerprint": "one-shot"}
            args.output.write_text(json.dumps(one_shot), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "run_fingerprint"):
                load_or_create_payload(args, spec)
            result = {
                "problem_id": "mmlu/test/x/0",
                "order": 0,
                "question": "q",
                "choices": CHOICES,
                "gold_answer": "C",
                "response": "Answer: C",
                "parsed_answer": None,
                "parse_method": "old_parser",
                "correct": False,
            }
            append_checkpoint(args.output, payload["run_fingerprint"], result)
            self.assertEqual(merge_checkpoint(args, spec, payload), 1)
            recovered = payload["benchmarks"]["mmlu"]["results"]
            self.assertEqual(len(recovered), 1)
            self.assertTrue(recovered[0]["correct"])
            self.assertEqual(recovered[0]["parsed_answer"], "C")
            self.assertEqual(merge_checkpoint(args, spec, payload), 0)
            self.assertEqual(len(recovered), 1)


if __name__ == "__main__":
    unittest.main()
