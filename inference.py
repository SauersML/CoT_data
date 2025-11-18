import os
import json
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

MODEL_NAME = "claude-sonnet-4-5"
MAX_TOKENS = 40000
THINKING_BUDGET_TOKENS = 20000

NUM_TRIALS_CONTROL = 300
NUM_TRIALS_EXPERIMENTAL_PHASE1 = 300
NUM_TRIALS_EXPERIMENTAL_PHASE2 = 300
MAX_PARALLEL_TRIALS = 50

OUTPUT_DIR = "sonnet_cot_experiment"
CONTEXT_PATH = "context_prompt.txt"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary.tsv")
TRIALS_JSONL_PATH = os.path.join(OUTPUT_DIR, "trials.jsonl")

GENERAL_PROMPT1_PATH = "general_prompt1.txt"
GENERAL_PROMPT2_PATH = "general_prompt2.txt"


UPPERCASE_50_PATTERN = re.compile(r"[A-Z]{50}")

client = anthropic.Anthropic()


def load_context_text():
    print(f"[load_context_text] Loading context from {CONTEXT_PATH}", flush=True)
    with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
        context = f.read()
    print(
        f"[load_context_text] Loaded {len(context)} characters of context text",
        flush=True,
    )
    return context


def build_phase1_prompt(condition, context_text, context_target_round):
    print(
        f"[build_phase1_prompt] Building phase 1 prompt for condition={condition}",
        flush=True,
    )
    with open(GENERAL_PROMPT1_PATH, "r", encoding="utf-8") as f:
        general_prompt1 = f.read()
    print(
        f"[build_phase1_prompt] General prompt 1 length: {len(general_prompt1)} characters",
        flush=True,
    )

    if context_target_round == "phase1":
        print(
            "[build_phase1_prompt] Appending context in phase 1",
            flush=True,
        )
        context_prompt = context_text
        combined = context_prompt + "\n\n" + general_prompt1
        print(
            f"[build_phase1_prompt] Combined prompt length: {len(combined)} characters",
            flush=True,
        )
        return combined

    print(
        f"[build_phase1_prompt] Returning control/unchanged prompt length: {len(general_prompt1)} characters",
        flush=True,
    )
    return general_prompt1


def build_phase2_prompt(condition, context_text, context_target_round):
    print(
        f"[build_phase2_prompt] Building phase 2 prompt for condition={condition}",
        flush=True,
    )
    with open(GENERAL_PROMPT2_PATH, "r", encoding="utf-8") as f:
        general_prompt2 = f.read()
    print(
        f"[build_phase2_prompt] Prompt length: {len(general_prompt2)} characters",
        flush=True,
    )

    if context_target_round == "phase2":
        print(
            "[build_phase2_prompt] Appending context in phase 2",
            flush=True,
        )
        context_prompt = context_text
        combined = context_prompt + "\n\n" + general_prompt2
        print(
            f"[build_phase2_prompt] Combined prompt length: {len(combined)} characters",
            flush=True,
        )
        return combined

    return general_prompt2


def join_thinking_blocks(content_blocks):
    print("[join_thinking_blocks] Joining thinking blocks", flush=True)
    parts = []
    for block in content_blocks:
        if block.type == "thinking":
            parts.append(block.thinking)
    total_len = sum(len(p) for p in parts)
    print(
        f"[join_thinking_blocks] Joined {len(parts)} thinking blocks into {total_len} characters",
        flush=True,
    )
    return "\n".join(parts)


def join_text_blocks(content_blocks):
    print("[join_text_blocks] Joining text blocks", flush=True)
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append(block.text)
    total_len = sum(len(p) for p in parts)
    print(
        f"[join_text_blocks] Joined {len(parts)} text blocks into {total_len} characters",
        flush=True,
    )
    return "\n".join(parts)


def extract_last_uppercase_50_sequence(text):
    print(
        "[extract_last_uppercase_50_sequence] Searching for 50-char uppercase sequences",
        flush=True,
    )
    if not text:
        print(
            "[extract_last_uppercase_50_sequence] Empty text; no sequences",
            flush=True,
        )
        return None
    matches = UPPERCASE_50_PATTERN.findall(text)
    if not matches:
        print(
            "[extract_last_uppercase_50_sequence] No 50-char uppercase sequences found",
            flush=True,
        )
        return None
    last_seq = matches[-1]
    print(
        f"[extract_last_uppercase_50_sequence] Found {len(matches)} sequences; returning last",
        flush=True,
    )
    return last_seq


def parse_output_string_and_metric(visible_text):
    print(
        "[parse_output_string_and_metric] Parsing visible text for String/Metric",
        flush=True,
    )
    guessed_string = None
    numeric_metric = None

    if not visible_text:
        print(
            "[parse_output_string_and_metric] No visible text provided",
            flush=True,
        )
        return None, None

    raw_lines = visible_text.splitlines()
    lines = []
    for raw in raw_lines:
        for segment in raw.split("\\n"):
            seg = segment.strip()
            if seg:
                lines.append(seg)

    print(
        f"[parse_output_string_and_metric] Visible text has {len(lines)} logical lines after expanding \\n",
        flush=True,
    )

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("String"):
            seq_match = UPPERCASE_50_PATTERN.search(stripped)
            if seq_match:
                guessed_string = seq_match.group(0)
            else:
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    guessed_string = parts[1].strip()
                else:
                    guessed_string = stripped[len("String") :].strip(" \t:-")
            print(
                "[parse_output_string_and_metric] Parsed String line",
                flush=True,
            )

        elif stripped.startswith("Metric") or stripped.startswith("Rating"):
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                metric_raw = parts[1].strip()
            else:
                keyword = "Metric" if stripped.startswith("Metric") else "Rating"
                metric_raw = stripped[len(keyword) :].strip(" \t:-")

            number_match = re.search(r"[-+]?\d+(\.\d+)?", metric_raw)
            if number_match:
                numeric_metric = number_match.group(0)
            else:
                numeric_metric = metric_raw
            print(
                "[parse_output_string_and_metric] Parsed Metric/Rating line",
                flush=True,
            )

    print(
        "[parse_output_string_and_metric] Result -> guessed_string length={}, metric={}".format(
            len(guessed_string) if guessed_string else 0, numeric_metric
        ),
        flush=True,
    )
    return guessed_string, numeric_metric


def compute_alignment(secret_sequence, guess_sequence):
    print(
        "[compute_alignment] Computing alignment between secret and guess sequences",
        flush=True,
    )
    if secret_sequence is None or guess_sequence is None:
        print(
            "[compute_alignment] Missing sequence(s); returning None alignment",
            flush=True,
        )
        return None, None, None
    if len(secret_sequence) != len(guess_sequence):
        print(
            "[compute_alignment] Sequence lengths differ; returning partial alignment",
            flush=True,
        )
        return None, len(secret_sequence), None
    matches = 0
    for a, b in zip(secret_sequence, guess_sequence):
        if a == b:
            matches += 1
    length = len(secret_sequence)
    score = matches / float(length)
    print(
        f"[compute_alignment] Alignment computed: matches={matches}, length={length}, score={score}",
        flush=True,
    )
    return matches, length, score


def ensure_output_dir():
    print(
        f"[ensure_output_dir] Ensuring output directory exists at {OUTPUT_DIR}",
        flush=True,
    )
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[ensure_output_dir] Created directory {OUTPUT_DIR}", flush=True)
    else:
        print(f"[ensure_output_dir] Directory {OUTPUT_DIR} already exists", flush=True)


def escape_for_tsv(value):
    if value is None:
        return ""
    return str(value).replace("\t", "\\t").replace("\n", "\\n")


def ensure_summary_header():
    print(
        f"[ensure_summary_header] Checking for summary TSV at {SUMMARY_PATH}",
        flush=True,
    )
    if not os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            header = [
                "condition",
                "context_target_round",
                "phase1_exact_i_understand",
                "phase1_secret_string",
                "phase2_guessed_string",
                "phase2_numeric_metric",
                "phase1_prompt",
                "phase1_thinking",
                "phase1_visible_text",
                "phase2_prompt",
                "phase2_thinking",
                "phase2_visible_text",
            ]
            f.write("\t".join(header) + "\n")
        print("[ensure_summary_header] Wrote TSV header", flush=True)
    else:
        print("[ensure_summary_header] Summary TSV already exists", flush=True)


def append_summary_row(trial_record):
    print("[append_summary_row] Appending trial record to summary TSV", flush=True)
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        condition = trial_record.get("condition", "")
        exact_flag = trial_record.get("phase1_exact_i_understand", False)
        phase1_secret = trial_record.get("secret_sequence")
        phase2_guess = trial_record.get("guessed_string")
        numeric_metric = trial_record.get("numeric_metric")

        fields = [
            escape_for_tsv(condition),
            escape_for_tsv(trial_record.get("context_target_round", "")),
            "True" if exact_flag else "False",
            escape_for_tsv(phase1_secret),
            escape_for_tsv(phase2_guess),
            escape_for_tsv(numeric_metric),
            escape_for_tsv(trial_record.get("phase1_prompt", "")),
            escape_for_tsv(trial_record.get("phase1_thinking", "")),
            escape_for_tsv(trial_record.get("phase1_visible_text", "")),
            escape_for_tsv(trial_record.get("phase2_prompt", "")),
            escape_for_tsv(trial_record.get("phase2_thinking", "")),
            escape_for_tsv(trial_record.get("phase2_visible_text", "")),
        ]
        f.write("\t".join(fields) + "\n")
    print("[append_summary_row] Summary TSV updated", flush=True)


def append_trial_jsonl(trial_record):
    print("[append_trial_jsonl] Appending trial record to JSONL log", flush=True)
    with open(TRIALS_JSONL_PATH, "a", encoding="utf-8") as f:
        json.dump(trial_record, f)
        f.write("\n")
    print("[append_trial_jsonl] JSONL log updated", flush=True)


def run_single_trial(trial_index, condition, context_text, context_target_round):
    print(
        f"[run_single_trial] Starting trial index={trial_index} condition={condition}",
        flush=True,
    )
    timestamp_iso = datetime.utcnow().isoformat() + "Z"
    print(f"[run_single_trial] Timestamp: {timestamp_iso}", flush=True)

    print(
        f"[run_single_trial] context_target_round resolved to {context_target_round}",
        flush=True,
    )

    phase1_prompt = build_phase1_prompt(
        condition, context_text, context_target_round
    )
    print(
        f"[run_single_trial] Phase 1 prompt length: {len(phase1_prompt)} characters",
        flush=True,
    )

    user_block = {
        "type": "text",
        "text": phase1_prompt,
    }

    conversation = [
        {
            "role": "user",
            "content": [user_block],
        }
    ]
    print("[run_single_trial] Conversation initialized with user prompt", flush=True)

    print("[run_single_trial] Sending phase 1 request to Anthropic", flush=True)
    response1 = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS},
        messages=conversation,
        timeout=600,
    )
    print(
        f"[run_single_trial] Phase 1 response received (id={response1.id}, model={response1.model})",
        flush=True,
    )

    phase1_thinking_text = join_thinking_blocks(response1.content)
    phase1_visible_text = join_text_blocks(response1.content)
    print(
        f"[run_single_trial] Phase 1 thinking length: {len(phase1_thinking_text)} characters",
        flush=True,
    )
    print(
        f"[run_single_trial] Phase 1 visible text length: {len(phase1_visible_text)} characters",
        flush=True,
    )

    phase1_exact_i_understand = phase1_visible_text.strip() == "I understand."
    print(
        f"[run_single_trial] Phase 1 exact 'I understand.': {phase1_exact_i_understand}",
        flush=True,
    )

    secret_sequence = extract_last_uppercase_50_sequence(phase1_thinking_text)
    print(
        f"[run_single_trial] Phase 1 secret sequence: {secret_sequence}",
        flush=True,
    )

    assistant_message_dict = response1.to_dict()
    assistant_content = assistant_message_dict["content"]

    conversation.append(
        {
            "role": "assistant",
            "content": assistant_content,
        }
    )
    print(
        "[run_single_trial] Appended full phase 1 response (thinking + text) to conversation",
        flush=True,
    )

    phase2_prompt = build_phase2_prompt(
        condition, context_text, context_target_round
    )
    print(
        f"[run_single_trial] Phase 2 prompt length: {len(phase2_prompt)} characters",
        flush=True,
    )
    conversation.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": phase2_prompt,
                }
            ],
        }
    )
    print("[run_single_trial] Added phase 2 user prompt to conversation", flush=True)

    print("[run_single_trial] Sending phase 2 request to Anthropic", flush=True)
    response2 = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS},
        messages=conversation,
        timeout=600,
    )
    print(
        f"[run_single_trial] Phase 2 response received (id={response2.id}, model={response2.model})",
        flush=True,
    )

    phase2_thinking_text = join_thinking_blocks(response2.content)
    phase2_visible_text = join_text_blocks(response2.content)
    print(
        f"[run_single_trial] Phase 2 thinking length: {len(phase2_thinking_text)} characters",
        flush=True,
    )
    print(
        f"[run_single_trial] Phase 2 visible text length: {len(phase2_visible_text)} characters",
        flush=True,
    )

    guessed_string, numeric_metric = parse_output_string_and_metric(phase2_visible_text)
    print(
        f"[run_single_trial] Phase 2 guessed string: {guessed_string}",
        flush=True,
    )
    print(
        f"[run_single_trial] Phase 2 numeric metric: {numeric_metric}",
        flush=True,
    )

    matches, length, score = compute_alignment(secret_sequence, guessed_string)
    print(
        f"[run_single_trial] Alignment results -> matches: {matches}, length: {length}, score: {score}",
        flush=True,
    )

    trial_record = {
        "timestamp_iso": timestamp_iso,
        "trial_index": trial_index,
        "condition": condition,
        "phase1_prompt": phase1_prompt,
        "phase2_prompt": phase2_prompt,
        "phase1_thinking": phase1_thinking_text,
        "phase1_visible_text": phase1_visible_text,
        "phase2_thinking": phase2_thinking_text,
        "phase2_visible_text": phase2_visible_text,
        "secret_sequence": secret_sequence,
        "guessed_string": guessed_string,
        "numeric_metric": numeric_metric,
        "phase1_exact_i_understand": phase1_exact_i_understand,
        "alignment_matches": matches,
        "alignment_length": length,
        "alignment_score": score,
        "response1_id": response1.id,
        "response2_id": response2.id,
        "response1_model": response1.model,
        "response2_model": response2.model,
        "context_target_round": context_target_round,
    }
    print("[run_single_trial] Trial record assembled", flush=True)

    return trial_record


def resolve_context_target_round(condition):
    if condition == "control":
        print(
            "[resolve_context_target_round] Control condition; context_target_round='control'",
            flush=True,
        )
        return "control"

    if condition == "experimental_phase1":
        print(
            "[resolve_context_target_round] Experimental phase1; context_target_round='phase1'",
            flush=True,
        )
        return "phase1"

    if condition == "experimental_phase2":
        print(
            "[resolve_context_target_round] Experimental phase2; context_target_round='phase2'",
            flush=True,
        )
        return "phase2"

    raise ValueError(
        "Unknown condition='{}'. Expected one of: 'control', 'experimental_phase1', 'experimental_phase2'.".format(
            condition
        )
    )


def main():
    print("[main] Starting experiment run", flush=True)
    ensure_output_dir()
    ensure_summary_header()
    context_text = load_context_text()
    print(
        f"[main] Context text length available for prompts: {len(context_text)} characters",
        flush=True,
    )

    trial_configs = []
    trial_index = 0
    for condition, num_trials in [
        ("control", NUM_TRIALS_CONTROL),
        ("experimental_phase1", NUM_TRIALS_EXPERIMENTAL_PHASE1),
        ("experimental_phase2", NUM_TRIALS_EXPERIMENTAL_PHASE2),
    ]:
        print(f"[main] Entering condition setup loop: {condition}", flush=True)
        for _ in range(num_trials):
            print(
                f"[main] Scheduling trial {trial_index} condition={condition}",
                flush=True,
            )
            target_round = resolve_context_target_round(condition)
            trial_configs.append((trial_index, condition, target_round))
            trial_index += 1

    print(
        f"[main] Total trials scheduled: {len(trial_configs)}; running with up to {MAX_PARALLEL_TRIALS} parallel trials",
        flush=True,
    )

    # No big in-memory trial_records; save as we go.
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TRIALS) as executor:
        futures = []
        for t_index, cond, context_target_round in trial_configs:
            futures.append(
                executor.submit(
                    run_single_trial,
                    t_index,
                    cond,
                    context_text,
                    context_target_round,
                )
            )

        # Persist each trial as soon as it finishes.
        for future in as_completed(futures):
            record = future.result()
            append_trial_jsonl(record)
            append_summary_row(record)
            print(
                f"[main] Persisted trial {record.get('trial_index')} condition={record.get('condition')}",
                flush=True,
            )

    print("[main] All trials completed and persisted", flush=True)


if __name__ == "__main__":
    main()

