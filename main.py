from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
from groq import Groq


class BiasAnalyzer:
    def __init__(self, model_path="./bias_detection_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        with open(f"{model_path}/label_encoder.pkl", "rb") as f:
            self.le = pickle.load(f)
    
    def analyze(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return {
            "prediction": self.le.inverse_transform([torch.argmax(probs).cpu().numpy()])[0],
            "probabilities": {label: float(prob) for label, prob in zip(self.le.classes_, probs[0])}
        }


def generate_fair_response(biased_text, detected_bias_categories):
    """Generate debiased summary response using Groq API"""
    client = Groq(api_key="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")
    
    # Construct targeted system message
    categories_str = ", ".join(detected_bias_categories)
    system_message = {
        "role": "system",
        "content": f"""You are a legal editor specializing in neutral summarization. Perform these tasks:
1. Remove {categories_str} bias completely
2. Summarize content while preserving legal meaning
3. Use balanced terminology for all parties
4. Avoid value judgments
5. Maintain factual accuracy
6. Structure with neutral framing

Apply these specific changes:
- Replace biased terms with neutral equivalents
- Remove emotional language
- Balance power dynamics in descriptions
- Use passive voice where appropriate
- Equalize party representations

Return ONLY the revised summary:"""
    }
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            system_message,
            {"role": "user", "content": biased_text}
        ],
        max_tokens=400,  # Increased for summarization
        temperature=0.2,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1
    )
    
    # Post-processing cleanup
    neutral_text = response.choices[0].message.content.strip()
    neutral_replacements = {
        "customer": "party",
        "corporation": "entity",
        "false claims": "disputed assertions",
        "selling": "transaction",
        "consumer": "individual"
    }
    
    for term, replacement in neutral_replacements.items():
        neutral_text = neutral_text.replace(term, replacement)
    
    return neutral_text




client = Groq(api_key="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")

def generate_legal_response(text: str) -> str:
    """Restructures legal text into standardized format"""
    prompt_template = f"""
    Restructure this legal content using the official template:
    {text}
    
    Required Format:
    1. Governing Law
    - Applicable Statutes: [Laws]
    - Amendments: [Changes]
    
    2. Key Definitions
    - [Term]: [Meaning]
    
    3. Legal Basis
    - Provisions: [Sections]
    - Thresholds: [Numbers]
    
    4. Dispute Resolution
    - Procedures: [Steps]
    - Forums: [Courts]
    
    5. Rights & Duties
    - Claimant: [Rights]
    - Respondent: [Obligations]
    
    6. Precedents
    - [Case]: [Principle]
    
    7. Remedies
    - Compensation: [Types]
    - Injunctions: [Options]
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal document restructurer. Format the provided text strictly using the template.\n"
                    "Rules:\n1. Preserve original meaning\n2. Use bullet points\n3. Extract key elements\n4. Maintain neutral language"
                )
            },
            {"role": "user", "content": prompt_template}
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()

def generate_tamil_translation(text: str) -> str:
    """Translates structured legal text to Tamil including sub-headings"""
    prompt_template = f"""
    Translate this entire legal document to Tamil while maintaining the exact structure:
    {text}
    
    Translation Rules:
    1. Translate all section headers to Tamil (e.g., "1. Governing Law" -> "1. ஆட்சிச் சட்டம்")
    2. Maintain original numbering and bullet points
    3. Preserve legal terminology accuracy
    4. Use proper Tamil Unicode formatting
    5. Keep the same document structure

    Example Header Translations:
    - Governing Law -> ஆட்சிச் சட்டம்
    - Key Definitions -> முக்கிய வரையறைகள்
    - Legal Basis -> சட்ட அடிப்படை
    - Dispute Resolution -> வழக்கு தீர்ப்பு
    - Rights & Duties -> உரிமைகள் & கடமைகள்
    - Precedents -> முன்னுதாரணங்கள்
    - Remedies -> தீர்வு முறைகள்
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal Tamil translator. Fully translate documents while maintaining structure and legal accuracy. Use standard Tamil legal terminology."
                },
                {"role": "user", "content": prompt_template}
            ],
            temperature=0.1,
            max_tokens=2500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Translation error: {str(e)}"

def refine_query(question):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal assistant. A user has asked a legal question:\n"
                    '"{question}"\n'
                    "Identify the jurisdiction, domain (e.g., tax, civil, criminal), and key legal terms.\n"
                    "Output only the key search terms and domain."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_tokens=200,  # Adjust response length if needed
    )
    
    return response.choices[0].message.content  # Extract model response

prompt_template = """
<s>[INST]
As a legal chatbot specializing in the Indian Penal Code, you are tasked with providing highly accurate and contextually appropriate responses. Ensure your answers meet these criteria:
- Respond in a bullet-point format to clearly delineate distinct aspects of the legal query.
- Each point should accurately reflect the breadth of the legal provision in question, avoiding over-specificity unless directly relevant to the user's query.
- Clarify the general applicability of the legal rules or sections mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
- Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
- Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations unless otherwise specified.
- Conclude with a brief summary that captures the essence of the legal discussion and corrects any common misinterpretations related to the topic.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
- [Detail the first key aspect of the law, ensuring it reflects general application]
- [Provide a concise explanation of how the law is typically interpreted or applied]
- [Correct a common misconception or clarify a frequently misunderstood aspect]
- [Detail any exceptions to the general rule, if applicable]
- [Include any additional relevant information that directly relates to the user's query]
</s>[INST]
"""









# if __name__ == "__main__":
#     analyzer = BiasAnalyzer()
#     while True:
#         text = input("\nEnter text to analyze (or 'quit' to exit): ")
#         if text.lower() == 'quit':
#             break
#         result = analyzer.analyze(text)
#         print(f"Predicted Bias: {result['prediction']}")
#         print("Confidence Scores:")
#         for label, score in result['probabilities'].items():
#             print(f"- {label}: {score:.2f}")