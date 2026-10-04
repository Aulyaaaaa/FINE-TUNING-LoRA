"""Demo UI Streamlit: chatbot "Sinar Elektronik" (Qwen2.5-0.5B-Instruct + adapter LoRA).

Jalankan:  streamlit run app.py
Syarat  :  folder adapter hasil training ada di ADAPTER_PATH (lihat README, bagian Demo UI).
"""
import os

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # harus sama dengan model saat training
ADAPTER_PATH = "qwen-sinar-elektronik-lora/final_adapter"

st.set_page_config(page_title="Asisten Sinar Elektronik")
st.title("Asisten Sinar Elektronik")
st.caption("Chatbot customer service hasil fine-tuning LoRA pada Qwen2.5-0.5B-Instruct")

if not os.path.isdir(ADAPTER_PATH):
    st.error(
        f"Folder adapter tidak ditemukan: `{ADAPTER_PATH}`.\n\n"
        "Jalankan notebook di Colab sampai Tahap 9, unduh folder `final_adapter`, "
        "lalu ekstrak ke lokasi tersebut (lihat README, bagian Demo UI)."
    )
    st.stop()


@st.cache_resource(show_spinner="Memuat model, mohon tunggu...")
def load_model():
    use_gpu = torch.cuda.is_available()
    dtype = torch.float16 if use_gpu else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.to("cuda" if use_gpu else "cpu")
    model.eval()
    return tokenizer, model


def tanya(tokenizer, model, pertanyaan, max_new_tokens=120):
    messages = [{"role": "user", "content": pertanyaan}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


tokenizer, model = load_model()

st.sidebar.header("Pengaturan")
pakai_lora = st.sidebar.checkbox("Pakai adapter LoRA", value=True)
st.sidebar.write(
    "Matikan centang untuk melihat jawaban model dasar sebagai pembanding "
    "sebelum fine-tuning."
)
if st.sidebar.button("Hapus riwayat"):
    st.session_state.riwayat = []

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

for peran, isi in st.session_state.riwayat:
    with st.chat_message(peran):
        st.write(isi)

if pesan := st.chat_input("Tulis pertanyaan kamu..."):
    st.session_state.riwayat.append(("user", pesan))
    with st.chat_message("user"):
        st.write(pesan)

    with st.chat_message("assistant"):
        with st.spinner("Menjawab..."):
            if pakai_lora:
                jawaban = tanya(tokenizer, model, pesan)
            else:
                with model.disable_adapter():
                    jawaban = tanya(tokenizer, model, pesan)
        st.write(jawaban)
    st.session_state.riwayat.append(("assistant", jawaban))
