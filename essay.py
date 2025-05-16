import streamlit as st
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from collections import Counter
import pandas as pd
import docx2txt  # Added for .docx parsing

# Downloading NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

# Loading API key
load_dotenv('key.env')

# Set up OpenAI API
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# Initializing session state
if "essay_text" not in st.session_state:
    st.session_state.essay_text = ""
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "essay_type" not in st.session_state:
    st.session_state.essay_type = ""
if "essay_prompt" not in st.session_state:
    st.session_state.essay_prompt = ""
if "max_word_count" not in st.session_state:
    st.session_state.max_word_count = 650
if "word_count_warning" not in st.session_state:
    st.session_state.word_count_warning = ""

def analyze_essay_structure(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    paragraphs = text.split('\n\n')
    
    # Performing basic counts
    word_count = len([word for word in words if word.isalnum()])
    sentence_count = len(sentences)
    avg_words_per_sentence = word_count / max(1, sentence_count)
    paragraph_count = len([p for p in paragraphs if p.strip()])
    
    # Now, sentence length 
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    sentence_length_variation = pd.Series(sentence_lengths).std() if sentence_lengths else 0
    
    # Next, we should analyze paragraphs
    intro = paragraphs[0] if paragraphs else ""
    conclusion = paragraphs[-1] if len(paragraphs) > 1 else ""
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_words_per_sentence": round(avg_words_per_sentence, 1),
        "sentence_length_variation": round(sentence_length_variation, 1),
        "intro_length": len(word_tokenize(intro)),
        "conclusion_length": len(word_tokenize(conclusion))
    }

def analyze_language_usage(text):
    words = word_tokenize(text.lower())
    word_freq = Counter([word for word in words if word.isalnum()])
    
    # Calculating unique words ratio
    unique_words = len(word_freq)
    total_words = sum(word_freq.values())
    unique_ratio = unique_words / max(1, total_words)
    
    # Checking for advanced transition words
    transition_words = ["however", "therefore", "consequently", "furthermore", "moreover",
                        "nevertheless", "alternatively", "similarly", "conversely", "indeed",
                        "specifically", "significantly", "ultimately", "meanwhile", "subsequently"]
    
    transition_count = sum(word_freq[word] for word in transition_words if word in word_freq)
    
    # Finding most common words
    simple_stop_words = ["the", "a", "an", "in", "on", "at", "to", "for", "and", "or", "but", "is", "are", 
                         "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", 
                         "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her", "its", 
                         "our", "their", "this", "that", "these", "those", "of", "with"]
    
    content_words = {word: count for word, count in word_freq.items() 
                     if word not in simple_stop_words and len(word) > 2}
    top_words = dict(sorted(content_words.items(), key=lambda x: x[1], reverse=True)[:10])
    
    return {
        "unique_words_ratio": round(unique_ratio, 2),
        "transition_words_count": transition_count,
        "top_words": top_words
    }

def analyze_with_ai(text, essay_type, prompt, max_word_count):
    
    system_prompt = """
    You are an expert essay reviewer with years of experience helping students improve their writing. 
    Provide a detailed, thoughtful analysis of this student essay. 
    
    Your analysis should include:
    
    1. Overall Assessment: General strengths and weaknesses
    2. Content Analysis: Depth, relevance, creativity, and insight
    3. Structure Analysis: Organization, flow, introduction, and conclusion
    4. Language Usage: Vocabulary, sentence variety, grammar, and style
    5. Specific Recommendations: Actionable suggestions for improvement
    6. Does essay answers to the prompt properly?
    7. Does the essay corresponds to the selected type? If no, then suggest some improvements so essay will correspond to its type.
    8. Scoring: Rate the essay on a scale of 1-10 in each category (Content, Structure, Language, Originality, Overall Impact)
    
    Make your feedback constructive, specific, and actionable. Provide examples from the text where helpful.
    """
    
    # Calculating word count for the prompt
    words = word_tokenize(text)
    word_count = len([word for word in words if word.isalnum()])
    
    # Adding word count info to the prompt
    word_count_info = ""
    if word_count > max_word_count:
        word_count_info = f"\nNOTE: This essay is {word_count} words, which exceeds the maximum limit of {max_word_count} words. Please address this in your feedback."
    
    user_prompt = f"""
    ESSAY TYPE: {essay_type}
    
    PROMPT: {prompt}
    
    WORD COUNT: {word_count} / Maximum {max_word_count}{word_count_info}
    
    ESSAY TEXT:
    {text}
    
    Please provide your comprehensive analysis of this essay.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        feedback = response.choices[0].message.content
        
        # Extracting scores using regex
        scores = {}
        score_pattern = r'(\w+):\s*(\d+(?:\.\d+)?)'
        matches = re.findall(score_pattern, feedback)
        for category, score in matches:
            if category.lower() in ['content', 'structure', 'language', 'originality', 'overall', 'impact', 'overall impact']:
                scores[category] = float(score)
        
        return {
            "feedback": feedback,
            "scores": scores
        }
    except Exception as e:
        return {
            "feedback": f"Error in AI analysis: {str(e)}",
            "scores": {}
        }

def get_improvement_suggestions(ai_feedback, structure_analysis, max_word_count):
    
    suggestions = []
    
    # Word count suggestions
    word_count = structure_analysis["word_count"]
    if word_count < 300:
        suggestions.append("Expand your essay with more detailed examples and explanations.")
    elif word_count > max_word_count:
        suggestions.append(f"Your essay exceeds the maximum word count of {max_word_count}. Trim your essay by removing redundant information or less important details.")
    
    # Paragraph structure suggestions
    paragraph_count = structure_analysis["paragraph_count"]
    if paragraph_count < 3:
        suggestions.append("Structure your essay with more paragraphs for better organization.")
    
    # Extracting suggestions from AI feedback
    ai_recommendation_patterns = [
        r"(?:I recommend|I suggest|You should|Consider|Try)([^.!?]*[.!?])",
        r"(?:could benefit from|would be better with|needs)([^.!?]*[.!?])"
    ]
    
    ai_suggestions = []
    for pattern in ai_recommendation_patterns:
        matches = re.findall(pattern, ai_feedback)
        ai_suggestions.extend(matches)
    
    # Cleaning up and adding AI suggestions
    for suggestion in ai_suggestions:
        suggestion = suggestion.strip()
        if suggestion and len(suggestion) > 15:
            if not suggestion.endswith("."):
                suggestion += "."
            suggestions.append(suggestion)
    
    # Limiting to top 5 most detailed suggestions
    suggestions = sorted(list(set(suggestions)), key=len, reverse=True)[:5]
    
    return suggestions

def check_word_count(text, max_count):
    words = word_tokenize(text)
    word_count = len([word for word in words if word.isalnum()])
    
    if word_count > max_count:
        return f"⚠️ WARNING: Your essay has {word_count} words, which exceeds the maximum limit of {max_count} words. Consider revising to meet the requirements."
    else:
        return f"Word count: {word_count} / {max_count} (within limit)"

def extract_text_from_docx(file):
    """Extract text from a .docx file"""
    try:
        # Using docx2txt to extract text from the Word document
        text = docx2txt.process(file)
        return text
    except Exception as e:
        return f"Error extracting text from document: {str(e)}"

def main():
    st.set_page_config(page_title="Essay Analyzer AI", layout="wide")
    
    st.title("Essay Analyzer AI")
    st.subheader("Get comprehensive feedback on your essays")
    
    # Sidebar for essay input
    with st.sidebar:
        st.header("Essay Information")
        
        essay_types = [
            "Personal Statement",
            "Argumentative Essay",
            "Diversity Essay",
            "Expository Essay", 
            "Narrative Essay",
            "Persuasive Essay",
            "Other"
        ]
        
        st.session_state.essay_type = st.selectbox(
            "Essay Type", 
            options=essay_types
        )
        
        st.session_state.essay_prompt = st.text_area(
            "Essay Prompt/Question", 
            help="Enter the original essay prompt or question"
        )
        
        # Adding maximum word count input
        st.session_state.max_word_count = st.number_input(
            "Maximum Word Count",
            min_value=100,
            max_value=5000,
            value=650,
            step=50,
            help="Set the maximum allowed word count for this essay"
        )
        
        # Essay input options
        input_method = st.radio("Input Method", ["Type/Paste Essay", "Upload File"])
        
        if input_method == "Type/Paste Essay":
            st.session_state.essay_text = st.text_area(
                "Paste your essay here", 
                height=300
            )
            
            # Adding up live word count check
            if st.session_state.essay_text:
                st.session_state.word_count_warning = check_word_count(
                    st.session_state.essay_text, 
                    st.session_state.max_word_count
                )
                st.markdown(st.session_state.word_count_warning)
                
        else:
            # Modified to accept only .docx files
            uploaded_file = st.file_uploader("Upload your essay (.docx)", type=["docx"])
            if uploaded_file:
                # Extract text from the .docx file
                st.session_state.essay_text = extract_text_from_docx(uploaded_file)
                
                # Adding word count check for uploaded file
                if st.session_state.essay_text:
                    st.session_state.word_count_warning = check_word_count(
                        st.session_state.essay_text, 
                        st.session_state.max_word_count
                    )
                    st.markdown(st.session_state.word_count_warning)
        
        # Submitting the button
        if st.button("Analyze Essay") and st.session_state.essay_text:
            with st.spinner("Analyzing your essay... This may take a minute."):
                # Getting structural analysis
                structure_analysis = analyze_essay_structure(st.session_state.essay_text)
                
                # Getting language analysis
                language_analysis = analyze_language_usage(st.session_state.essay_text)
                
                # Getting AI feedbacks
                ai_analysis = analyze_with_ai(
                    st.session_state.essay_text,
                    st.session_state.essay_type,
                    st.session_state.essay_prompt,
                    st.session_state.max_word_count
                )
                
                # Generating improvement suggestions
                improvement_suggestions = get_improvement_suggestions(
                    ai_analysis["feedback"],
                    structure_analysis,
                    st.session_state.max_word_count
                )
                
                # Storing theresults
                st.session_state.analysis_results = {
                    "structure": structure_analysis,
                    "language": language_analysis,
                    "ai_feedback": ai_analysis["feedback"],
                    "scores": ai_analysis["scores"],
                    "suggestions": improvement_suggestions
                }
    
    if st.session_state.analysis_results:
        # Creating tabs for different analysis views
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Detailed Feedback", "Structure Analysis", "Improvement Plan"])
        
        with tab1:
            st.header("Essay Analysis Overview")
            
            # Displaying essay metadata
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Essay Information")
                st.write(f"**Type:** {st.session_state.essay_type}")
                if st.session_state.essay_prompt:
                    st.write(f"**Prompt:** {st.session_state.essay_prompt}")
                
                word_count = st.session_state.analysis_results['structure']['word_count']
                max_count = st.session_state.max_word_count
                
                if word_count > max_count:
                    st.error(f"**Word Count:** {word_count} / {max_count} (exceeds limit by {word_count - max_count} words)")
                else:
                    st.success(f"**Word Count:** {word_count} / {max_count}")
            
            with col2:
                st.subheader("Quality Scores")
                scores = st.session_state.analysis_results["scores"]
                
                if scores:
                    # Displaying scores as text instead of chart
                    for category, score in scores.items():
                        st.write(f"**{category}:** {score}/10")
                else:
                    st.write("No scores available.")
            
            # Quick summary of feedback
            st.subheader("Summary Feedback")
            ai_feedback = st.session_state.analysis_results["ai_feedback"]
            paragraphs = ai_feedback.split('\n\n')
            if paragraphs:
                st.write(paragraphs[0])
                if len(paragraphs) > 1:
                    st.write(paragraphs[1])
            
            st.subheader("Key Recommendations")
            suggestions = st.session_state.analysis_results["suggestions"]
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
        
        with tab2:
            st.header("Detailed Essay Feedback")
            st.write(st.session_state.analysis_results["ai_feedback"])
            
            # Displaying essay with expandable option
            with st.expander("View Your Essay"):
                st.write(st.session_state.essay_text)
        
        with tab3:
            st.header("Essay Structure and Language Analysis")
            
            structure = st.session_state.analysis_results["structure"]
            language = st.session_state.analysis_results["language"]
            
            # Displaying structure metrics
            st.subheader("Structure Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                word_count = structure["word_count"]
                max_count = st.session_state.max_word_count
                
                if word_count > max_count:
                    st.error(f"Word Count: {word_count} / {max_count}")
                elif word_count < 300:
                    st.warning(f"Word Count: {word_count} / {max_count}")
                else:
                    st.success(f"Word Count: {word_count} / {max_count}")
            with col2:
                st.write(f"**Paragraphs:** {structure['paragraph_count']}")
            with col3:
                st.write(f"**Sentences:** {structure['sentence_count']}")
            with col4:
                st.write(f"**Avg Sentence Length:** {structure['avg_words_per_sentence']}")
            
            # Language analysis
            st.subheader("Language Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Unique Words Ratio:** {language['unique_words_ratio']*100:.1f}%")
                st.write(f"**Transition Words:** {language['transition_words_count']}")
            
            with col2:
                st.subheader("Most Used Words")
                for word, count in language["top_words"].items():
                    st.write(f"**{word}**: {count}")
            
            # Paragraph breakdown
            st.subheader("Essay Flow")
            paragraphs = [p for p in st.session_state.essay_text.split('\n\n') if p.strip()]
            
            if len(paragraphs) >= 2:
                with st.expander("Introduction"):
                    st.write(paragraphs[0])
                
                if len(paragraphs) > 2:
                    with st.expander("Body Paragraphs"):
                        for i, para in enumerate(paragraphs[1:-1], 1):
                            st.write(f"**Paragraph {i+1}:**")
                            st.write(para)
                            st.write("---")
                
                with st.expander("Conclusion"):
                    st.write(paragraphs[-1])
            else:
                st.warning("Your essay doesn't have enough clearly defined paragraphs. Consider restructuring with clear paragraph breaks.")
        
        with tab4:
            st.header("Improvement Plan")
            
            # Displaying all improvement suggestions
            st.subheader("Specific Recommendations")
            for i, suggestion in enumerate(st.session_state.analysis_results["suggestions"], 1):
                st.write(f"{i}. {suggestion}")
            
            # Areas of focus based on scores
            st.subheader("Focus Areas")
            scores = st.session_state.analysis_results["scores"]
            if scores:
                lowest_score = min(scores.items(), key=lambda x: x[1])
                highest_score = max(scores.items(), key=lambda x: x[1])
                
                st.write(f"**Priority for Improvement:** {lowest_score[0]} (Score: {lowest_score[1]})")
                st.write(f"**Strongest Element:** {highest_score[0]} (Score: {highest_score[1]})")
                
                # General advice based on lowest score category
                advice = {
                    "Content": "Focus on adding more depth, examples, and unique insights to strengthen your content.",
                    "Structure": "Work on improving your essay's organization, transitions between paragraphs, and overall flow.",
                    "Language": "Enhance your vocabulary usage, sentence variety, and overall writing style.",
                    "Originality": "Develop more unique perspectives and avoid common clichés or overly generic approaches.",
                    "Overall": "Consider revising multiple aspects of your essay following the specific suggestions above.",
                    "Impact": "Make your essay more memorable by strengthening your personal voice and key messages.",
                    "Overall Impact": "Focus on making your main points clearer and more compelling to increase the essay's impact."
                }
                
                category = lowest_score[0]
                if category in advice:
                    st.write(advice[category])
            
            # Next steps checklist
            st.subheader("Next Steps Checklist")
            st.write("""
            - [ ] Review the detailed feedback thoroughly
            - [ ] Make notes on the specific areas needing improvement
            - [ ] Revise your essay focusing on the priority areas
            - [ ] Check for grammar and spelling errors
            - [ ] Read your essay aloud to check flow and clarity
            - [ ] Get feedback from a teacher or peer
            - [ ] Submit your revised essay for another analysis
            """)
    else:
        # Showing welcome info
        st.write("""
        ## Welcome to the Essay Analyzer AI
        
        **How it works:**
        1. Select your essay type in the sidebar
        2. Enter the essay prompt (if applicable)
        3. Set the maximum word count for your essay
        4. Type or upload your essay (.docx format)
        5. Click "Analyze Essay" to get comprehensive feedback
        
        Our AI system will analyze your essay's structure, language, content, and effectiveness,
        then provide detailed feedback to help you improve.
        
        **Benefits:**
        - Get instant, detailed feedback on your writing
        - Check if your essay meets the word count requirements
        - Identify specific strengths and weaknesses
        - Receive actionable suggestions for improvement
        - Understand how your essay scores across multiple dimensions
        - Support for Microsoft Word (.docx) files
        
        **Ready to improve your essay?** Add your essay details in the sidebar to begin.
        """)

if __name__ == "__main__":
    main()