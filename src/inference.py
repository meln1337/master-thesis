import torch
import gradio as gr
import nltk

from model import load_detector

nltk.download("punkt_tab", quiet=True)

WINDOW_SIZE = 3
MAX_LENGTH = 768
CHECKPOINT = "weights/ai-text-detector"

model, tokenizer, device = None, None, None


def ensure_model_loaded():
    global model, tokenizer, device
    if model is None:
        model, tokenizer, device = load_detector(CHECKPOINT)


def score_text(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=MAX_LENGTH, padding=True,
    ).to(device)
    with torch.no_grad():
        logit = model(**inputs)
    return torch.sigmoid(logit).item()


def analyze_with_sliding_window(text):
    ensure_model_loaded()
    sentences = nltk.sent_tokenize(text)

    if len(sentences) == 0:
        return 0.0, "<p>No text provided.</p>"

    if len(sentences) == 1:
        score = score_text(sentences[0])
        color = score_to_color(score)
        html = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{sentences[0]}</span>'
        return score, html

    sentence_scores = [0.0] * len(sentences)
    sentence_counts = [0] * len(sentences)

    for i in range(len(sentences)):
        start = max(0, i - WINDOW_SIZE // 2)
        end = min(len(sentences), start + WINDOW_SIZE)
        if end - start < WINDOW_SIZE:
            start = max(0, end - WINDOW_SIZE)

        window_text = " ".join(sentences[start:end])
        window_score = score_text(window_text)

        for j in range(start, end):
            sentence_scores[j] += window_score
            sentence_counts[j] += 1

    for i in range(len(sentences)):
        if sentence_counts[i] > 0:
            sentence_scores[i] /= sentence_counts[i]

    overall_score = sum(sentence_scores) / len(sentence_scores)

    html_parts = []
    for sentence, score in zip(sentences, sentence_scores):
        color = score_to_color(score)
        tooltip = f"AI probability: {score:.1%}"
        html_parts.append(
            f'<span title="{tooltip}" style="background-color: {color}; '
            f'padding: 2px 4px; border-radius: 3px; margin: 1px;">'
            f'{sentence}</span>'
        )
    html_output = " ".join(html_parts)

    return overall_score, html_output


def score_to_color(score):
    if score > 0.7:
        return f"rgba(239, 68, 68, {min(score, 0.9)})"
    elif score > 0.4:
        return f"rgba(234, 179, 8, {min(score + 0.2, 0.9)})"
    else:
        return f"rgba(34, 197, 94, {max(0.3, 1 - score)})"


def detect(text, threshold):
    overall_score, highlighted_html = analyze_with_sliding_window(text)

    if overall_score > threshold:
        verdict = f"AI-generated (confidence: {overall_score:.1%})"
    else:
        verdict = f"Human-written (confidence: {1 - overall_score:.1%})"

    legend = """
    <div style="margin-top: 12px; font-size: 14px;">
        <span style="background-color: rgba(239, 68, 68, 0.7); padding: 2px 8px; border-radius: 3px;">AI-generated</span>
        <span style="background-color: rgba(234, 179, 8, 0.7); padding: 2px 8px; border-radius: 3px; margin-left: 8px;">Uncertain</span>
        <span style="background-color: rgba(34, 197, 94, 0.5); padding: 2px 8px; border-radius: 3px; margin-left: 8px;">Human-written</span>
    </div>
    """

    full_html = f"""
    <div style="line-height: 1.8; font-size: 16px; padding: 12px;">
        {highlighted_html}
    </div>
    {legend}
    """

    return verdict, f"{overall_score:.4f}", full_html


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Textbox(
            label="Input Text",
            placeholder="Paste text here to check if it was written by AI...",
            lines=10,
        ),
        gr.Slider(
            minimum=0.1, maximum=0.9, value=0.5, step=0.05,
            label="Detection Threshold",
        ),
    ],
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Textbox(label="AI Probability"),
        gr.HTML(label="Sentence-Level Analysis"),
    ],
    title="AI Text Detector",
    description="Detects AI-generated text using sentence-level analysis with a sliding window approach.",
    examples=[
        ["The quick brown fox jumps over the lazy dog. This is a simple test sentence. Natural language is beautiful.", 0.5],
    ],
)


if __name__ == "__main__":
    demo.launch()
