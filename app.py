import streamlit as st
import tempfile
from dspy_qa import DocQA

st.set_page_config(page_title="DoC QA", layout="wide")
# st.title("📄 Chat over PDF using DSPy 🚀")
st.markdown("""
<div class="st-emotion-cache-12fmjuu">
   <h1>Document QA DSPY</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.st-emotion-cache-12fmjuu {


  text-align: center;
}

  margin: 0;
  font-size: 24px;
  justify-content: center;

}

.st-af st-b6 st-b7 st-ar st-as st-b8 st-b9 st-ba st-bb st-bc st-bd st-be st-b1{
  color:black;
}
</style>
""", unsafe_allow_html=True)

# Initialize chat history
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if st.session_state.knowledge_base is None:
    pdf_file = st.file_uploader("Upload a PDF file", "pdf")
    if pdf_file:
      with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
          temp_file.write(pdf_file.getbuffer())
          temp_file_path = temp_file.name
          docqa = DocQA(temp_file_path)
          if docqa:
              st.success("PDF file uploaded successfully!")

          st.session_state.knowledge_base = docqa

docqa = st.session_state.knowledge_base
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if st.button("Reset"):
        st.session_state.knowledge_base = None
        st.session_state.messages = []
        st.rerun()  # Rerun the script


# Accept user input
if question := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
          st.markdown(question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = docqa.run(question)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})