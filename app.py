import gradio as gr
import numpy as np

from transformers import pipeline

# caption = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
caption = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# generate = pipeline("text-generation", model="microsoft/phi-2")
generate = pipeline("text-generation", model="openai-community/gpt2")

tts = pipeline(task="text-to-speech", model="facebook/mms-tts-eng")

def run_caption(img):
  res = caption(img, max_new_tokens=128)
  return res[0]["generated_text"]

def run_generate(txt):
  res = generate(txt, max_length=50)
  return res[0]["generated_text"]

def run_tts(txt):
  res = tts(txt)
  audio = (res["audio"].reshape(-1) * 2 ** 15).astype(np.int16)
  return res["sampling_rate"], audio


def run_caption_tts(img):
  return run_tts(run_caption(img))

def run_caption_generate_tts(img):
  return run_tts(run_generate(run_caption(img)))


with gr.Blocks() as demo:
  gr.Interface(
    run_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
  )

  gr.Interface(
    run_generate,
    inputs="text",
    outputs="text",
  )

  gr.Interface(
    run_tts,
    inputs=gr.Textbox(),
    outputs="audio",
  )

  gr.Interface(
    run_caption_tts,
    inputs=gr.Image(type="pil"),
    outputs="audio",
  )

  gr.Interface(
    run_caption_generate_tts,
    inputs=gr.Image(type="pil"),
    outputs="audio",
  )

if __name__ == "__main__":
   demo.launch()
