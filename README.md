# banking-intent-classifier

A fine-tuned Qwen3-0.6B model that classifies bank customer messages into 77 intents (Banking77). It is trained with LoRA using Unsloth and compared against Gemini on the same held-out messages.

## Results

Both models were scored on the same 100 test messages with exact-match grading.

| Model | Accuracy | Intent list in prompt | Cost |
|---|---|---|---|
| Qwen3-0.6B + LoRA | 86% | No | $0 (free Colab T4) |
| gemini-3.1-flash-lite | 77% | Yes, all 77 | $0 on free tier (56,468 input and 514 output tokens) |

The sample is small, so the margin of error is about 7 points.

## Requirements

- Google Colab with a T4 GPU (Runtime > Change runtime type)
- Python packages: `unsloth`, `datasets`, `google-genai`
- Optional: a Gemini API key 
## Setup and run

Open `banking-intent-classifier.ipynb` in Colab and run the cells in order. The install cell is:

```
pip install unsloth datasets google-genai
```

The notebook then:

1. Loads `PolyAI/banking77` and converts each row into a chat (message in, intent name out).
2. Loads `unsloth/Qwen3-0.6B` and adds a LoRA adapter (r = 16).
3. Trains for 300 steps, computing loss on the assistant's answer only.
4. Scores the model on 100 test messages (test split shuffled with seed 42).
5. Scores gemini-3.1-flash-lite on the same 100 messages.

## Evaluation

The fine-tuned model gets only the customer message. Gemini gets the message plus the full list of 77 intents in its system prompt, which is about 565 input tokens per request.

An answer counts as correct only if it exactly matches the intent name. Qwen3 emits an empty `<think></think>` block before its answer, which is stripped before comparing. Without that step, the score was 0%.

## Limitations

- 100 test messages only, because of Gemini free-tier rate limits.
- Gemini was prompted zero-shot with no examples, so other prompts could change its score.
- Exact-match grading can mark a correct answer wrong if the wording differs slightly.

## Next steps

Rerun both models on all 3,080 test messages.
