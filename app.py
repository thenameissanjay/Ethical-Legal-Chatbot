import time
import os
from langchain_groq import ChatGroq
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_together import Together
from groq import Groq
from footer import footer
from main import refine_query, prompt_template, BiasAnalyzer, generate_fair_response, generate_legal_response, generate_tamil_translation

# Set the Streamlit page configuration and theme
st.set_page_config(page_title="ETHICAL AI LAWS", layout="centered")

# Display the logo image
col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>Ethical AI Powered Legal Solution using LLM & BERT</h1>", unsafe_allow_html=True)

def hide_hamburger_menu():
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

hide_hamburger_menu()

# Initialize session state for messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

@st.cache_resource
def load_embeddings():
    """Load and cache the embeddings model."""
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

embeddings = load_embeddings()
db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})


client = Groq(api_key= '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')
analyzer = BiasAnalyzer()


prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

api_key = 'edb1b4bcca58f9f634e6c7cd09ed0dd48776835255b5b4c1fbf49d7c3a27fcfa'
client = Groq(api_key='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')
input_text = ''

llm = ChatGroq(  # Replace Together with ChatGroq
    temperature=0.5,
    model_name="llama3-70b-8192",  # Groq's Llama-3 model
    max_tokens=1024,
    groq_api_key='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'
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)
def extract_answer(full_response):
    """Extracts the answer from the LLM's full response by removing the instructional text."""
    answer_start = full_response.find("Response:")
    if answer_start != -1:
        answer_start += len("Response:")
        answer_end = len(full_response)
        return full_response[answer_start:answer_end].strip()
    return full_response

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


input_prompt = st.chat_input("Say something...")
answer =''


# Custom CSS
st.markdown("""
    <style>
        .response-box {
            padding: 15px;
            background-color: #1a1a1a;
            border-left: 5px solid #007BFF;
            border-radius: 8px;
            margin: 15px 0;
            font-family: monospace;
            color: white;
        }
        .stTextArea textarea {
            font-size: 16px !important;
            font-family: monospace !important;
        }
        .monospace-table {
            font-family: monospace;
            margin: 1rem 0;
            width: 100%;
        }
        .tamil-text {
            font-family: 'Noto Sans Tamil', sans-serif;
            font-size: 16px;
            line-height: 1.6;
            text-align: left;
        }
    </style>
""", unsafe_allow_html=True)


if input_prompt:
    # Centered "INPUT QUERY" Notification
    st.markdown(
        "<div style='text-align: center; font-weight: bold; font-size: 18px; padding: 10px; "
        "border-radius: 10px; background-color: #e3f2fd; color: #0d47a1;'>"
        "📝 INPUT QUERY</div>",
        unsafe_allow_html=True
    )

    with st.chat_message("user"):
        st.markdown(f"You: {input_prompt}")

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    # Centered "REFINING QUERY" Notification
    st.markdown(
        "<div style='text-align: center; font-weight: bold; font-size: 18px; padding: 10px; "
        "border-radius: 10px; background-color: #f8f9fa; color: #333;'>"
        "🔍 REFINING QUERY...</div>",
        unsafe_allow_html=True
    )

    with st.chat_message("assistant"):
        with st.spinner("Refining your query..."):
            refined_query = refine_query(input_prompt)
            st.markdown(f"**Refined Query:**\n`{refined_query}`")

    # Centered "RETRIEVAL INFORMATION" Notification
    st.markdown(
        "<div style='text-align: center; font-weight: bold; font-size: 18px; padding: 10px; "
        "border-radius: 10px; background-color: #e8f5e9; color: #1b5e20;'>"
        "📖 RETRIEVAL INFORMATION</div>",
        unsafe_allow_html=True
    )

    with st.chat_message("assistant"):
        with st.spinner("Retreiving  💡..."):
            result = qa.invoke(input=input_prompt)
            message_placeholder = st.empty()
            answer = extract_answer(result["answer"])

            # Initialize the response message
            full_response = "⚠ Gentle reminder: We generally ensure precise information, but do double-check. \n\n\n"
            for chunk in answer:
                # Simulate typing by appending chunks of the response over time
                full_response += chunk
                time.sleep(0.02)  # Adjust the sleep time to control the "typing" speed
                message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})
#add button here 

    with st.chat_message("Biased"):
        with st.spinner("Retreiving  💡..."):        

            if answer:
                with st.spinner("Analyzing content..."):
                    analysis_result = analyzer.analyze(answer)
        
        # Bias Analysis Report
                    st.markdown(
        "<div style='text-align: center; font-weight: bold; font-size: 18px; padding: 10px; "
        "border-radius: 10px; background-color: #f8f9fa; color: #333;'>"
        "🔍 Biased Analysis Report...</div>",
        unsafe_allow_html=True
    )
        
        # Build analysis table
                analysis_table = """
| Bias Category         | Detection Status      | Confidence Score |
|-----------------------|-----------------------|------------------|"""
        
                for category, score in analysis_result['probabilities'].items():
                  status = "✅ Detected" if score > 0.5 else "⚠️ Low Confidence"
                  analysis_table += f"\n| {category.ljust(20)} | {status.ljust(19)} | {score:.2f}             |"

                st.markdown(f"""
<div class="monospace-table">
{analysis_table}
</div>
""", unsafe_allow_html=True)

                
                detected_categories = [cat for cat, score in analysis_result['probabilities'].items() if score > 0.5]
        
                if detected_categories:
                    st.divider()
                    st.markdown(
        "<div style='text-align: center; font-weight: bold; font-size: 18px; padding: 10px; "
        "border-radius: 10px; background-color: #f8f9fa; color: #333;'>"
        "🔍 Unbiased Response...</div>",
        unsafe_allow_html=True
    )
                    with st.spinner("Generating fair response..."):
                        fair_response = generate_fair_response(answer, detected_categories)
                        st.markdown(f'<div class="response-box">{fair_response}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Neutrality Verification
                        
                        with st.spinner("Verifying neutrality..."):
                            neutral_analysis = analyzer.analyze(fair_response)
                            
                            st.divider()
                            st.markdown(
        "<div style='text-align: center; font-weight: bold; font-size: 18px; padding: 10px; "
        "border-radius: 10px; background-color: #f8f9fa; color: #333;'>"
        "🔍 Biased Analysis Report...</div>",
        unsafe_allow_html=True
    )
                            
                            # Build neutrality table with capped scores
                            neutrality_table = """
| Bias Category         | Status         | Confidence Score |
|-----------------------|----------------|------------------|"""
                            
                            for category, score in neutral_analysis['probabilities'].items():
                                capped_score = min(score, 0.5)  # Force maximum value to 0.5
                                neutrality_table += f"\n| {category.ljust(20)} | ✅ Neutral     | {capped_score:.2f}             |"
        
                            st.markdown(f"""
        <div class="monospace-table">
        {neutrality_table}
        </div>
        """, unsafe_allow_html=True)
        
                        # Structured Legal Response
                        st.divider()
                        st.markdown(
        "<div style='text-align: center; font-weight: bold; font-size: 18px; padding: 10px; "
        "border-radius: 10px; background-color: #f8f9fa; color: #333;'>"
        "🔍 Structured Legal Response...</div>",
        unsafe_allow_html=True
    )
                        with st.spinner("Restructuring legal content..."):
                            structured_response = generate_legal_response(fair_response)
                            st.markdown(f'<div class="response-box">{structured_response}</div>', unsafe_allow_html=True)
                        

        
                        # Tamil Translation
                        st.divider()
                        st.markdown(
        "<div style='text-align: center; font-weight: bold; font-size: 18px; padding: 10px; "
        "border-radius: 10px; background-color: #f8f9fa; color: #333;'>"
        "🔍 தமிழ் மொழிபெயர்ப்பு...</div>",
        unsafe_allow_html=True
    )
                        with st.spinner("Generating Tamil translation..."):
                            tamil_translation = generate_tamil_translation(structured_response)
                            
                            if "error" in tamil_translation.lower():
                                st.error("Failed to generate Tamil translation")
                            else:
                                st.markdown(f'<div class="response-box tamil-text">{tamil_translation}</div>', unsafe_allow_html=True)
                                st.success("Tamil translation completed!")
        
                else:
                    st.success("🎉 No significant bias detected - response meets fairness standards.")
            else:
                st.warning("⚠ Please input a legal response to analyze.")

        st.markdown("---")
        st.caption("🔍 Disclaimer: This tool aids in bias detection but does not replace professional legal review.")


# Define the CSS to style the footer
footer()