import streamlit as st
from streamlit_ace import st_ace
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.header("Question Answering over Code. With Summaries.")

DEVICE = 'cuda:0'

@st.cache_resource
def get_model():
    checkpoint = "../starcoder/bigcode/Starcoder1024Tokens32LoraRankSummaries/starcoder-merged"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

@st.cache_resource
def get_summary_model():
    summary_model_path = '/home/user/vladimir/code-sum/fine-tuning/CodeT5/CodeT5+/saved_models/funcom-220m-py/checkpoint-92533'
    summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_path)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_path, device_map=DEVICE)
    return summary_tokenizer, summary_model

qa_tokenizer, qa_model = get_model()
summary_tokenizer, summary_model = get_summary_model()

with st.form(key="my_form"):
    
    code = st_ace(
        auto_update=True,
        language='Python',
        value='''def group_id_or_name_exists(reference, context):
    model = context['model']
    result = model.Group.get(reference)
    if (not result):
        raise Invalid(_('That group name or ID does not exist.'))
    return reference
'''
    )

    # question = st.text_area(
    #     "Write your question about this code",
    #     "What does this code do?",
    #     height=25,
    #     help="Write your question about this code",
    #     key="2",
    # )

    question = st.text_input(
        label="Write your question about this code",
        value="What does this code do?"
    )
    
    submit_button = st.form_submit_button(label="Ask")


if submit_button and not code:
    st.warning("There is no code")
    st.stop()

if submit_button and not question:
    st.warning("There is no question")
    st.stop()

elif submit_button:

    start_time = time.time()

    # generate summary
    summary_inputs = summary_tokenizer(code, return_tensors="pt").to(DEVICE)
    summary_outputs = summary_model.generate(**summary_inputs, max_new_tokens=128)
    summary = summary_tokenizer.decode(summary_outputs[0], skip_special_tokens=True)
    
    prompt = f"Question: {question}\n\nSummary: {summary}\n\nCode: {code}\n\nAnswer:"
    inputs = qa_tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    outputs = qa_model.generate(inputs, pad_token_id=qa_tokenizer.pad_token_id, max_new_tokens=256)

    answer = qa_tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False)
    answer = answer.replace(prompt, "").strip()
    answer = answer[:-13]

    st.write(f"Generated Summary: {summary}")
    
    st.markdown(f"""#### Answer:
```
    {answer}
```""")

    st.text(f"Running time: {round(time.time() - start_time, 2)} seconds")
