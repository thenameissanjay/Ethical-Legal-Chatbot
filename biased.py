import torch
import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from groq import Groq
from main import BiasAnalyzer, generate_fair_response, generate_legal_response, generate_tamil_translation

# Initialize Bias Analyzer
analyzer = BiasAnalyzer()

client = Groq(api_key="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")

# Streamlit UI Configuration
st.set_page_config(page_title="Legal Bias Analyzer", layout="centered")
st.title("⚖️ Legal Bias Analyzer")
st.markdown("### Detect and Mitigate Bias in Legal Responses")

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



# Input area for legal text
input_text = st.text_area("Paste legal response here:", 
                         height=150, 
                         placeholder="Enter legal text to analyze...")

if st.button("Analyze & Mitigate Bias", use_container_width=True):
    if input_text.strip():
        with st.spinner("Analyzing content..."):
            analysis_result = analyzer.analyze(input_text)

        # Bias Analysis Report
        st.subheader("📊 Bias Analysis Report")
        
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
            st.subheader("🔄 Unbiased Legal Response")
            with st.spinner("Generating fair response..."):
                fair_response = generate_fair_response(input_text, detected_categories)
                st.markdown(f'<div class="response-box">{fair_response}</div>', 
                           unsafe_allow_html=True)
                
                # Neutrality Verification
                
                with st.spinner("Verifying neutrality..."):
                    neutral_analysis = analyzer.analyze(fair_response)
                    
                    st.divider()
                    st.subheader("✅ Neutrality Verification Report")
                    
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
                    st.success("Full neutrality achieved through debiasing process!")

                # Structured Legal Response
                st.divider()
                st.subheader("📄 Structured Legal Response")
                with st.spinner("Restructuring legal content..."):
                    structured_response = generate_legal_response(fair_response)
                    st.markdown(f'<div class="response-box">{structured_response}</div>', unsafe_allow_html=True)
                
                st.success("Legal content successfully restructured!")

                # Tamil Translation
                st.divider()
                st.subheader("🌐 Tamil Version")
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
        st.warning("⚠️ Please input a legal response to analyze.")

st.markdown("---")
st.caption("🔍 Disclaimer: This tool aids in bias detection but does not replace professional legal review.")