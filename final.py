import os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint
import requests

# Streamlit APP Configuration
st.set_page_config(page_title="Summarize Text", page_icon="🦜", layout="centered")

# Page Title and Subtitle
st.title("Chat-Mate... Summarize Text From YT or Website using HuggingFace 📝")
st.subheader('Summarize any URL with ease')

# Input HuggingFace API key
hf_api_key = st.text_input("Enter your HuggingFace API Token", value="", type="password")

# URL input and language selection
generic_url = st.text_input("Enter the URL (YouTube video or website)")
selected_language = st.selectbox("Choose Transcript Language", ["English (en)", "Hindi (hi)"])  # Language options

# Template for summary prompt
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarize Button
if st.button("Summarize Now"):
    # Validate inputs
    if not hf_api_key.strip():
        st.error("Please provide a HuggingFace API token.")
    elif not generic_url.strip():
        st.error("Please enter a URL to proceed.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It should be a YouTube video or a website URL.")
    else:
        try:
            # Initialize the HuggingFace model after user provides the API key
            repo_id = "mistralai/Mistral-7B-Instruct-v0.3"  # Replace with the desired model
            llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=hf_api_key)

            with st.spinner("Processing..."):
                # Load data from YouTube or website
                try:
                    if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True, language=selected_language[:2])
                        docs = loader.load()
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                        )
                        docs = loader.load()

                    # Provide a dropdown to view the extracted content
                    if docs:
                        with st.expander("View Extracted Content"):
                            st.write(docs[0].page_content)
                    else:
                        st.error("No content could be extracted from the provided URL. Please check if the URL is correct and accessible.")
                
                except Exception as e:
                    st.error(f"Error loading content from the URL: {e}")

                # Summarization chain
                try:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    st.success("Summary:")
                    st.write(output_summary)

                    # Download the summary as a text file
                    if output_summary:
                        st.download_button(
                            label="Download Summary",
                            data=output_summary,
                            file_name="summary.txt",
                            mime="text/plain",
                        )
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        st.error("Rate limit exceeded. Please wait a moment and try again or use a different API key.")
                    else:
                        st.exception(f"HTTP Error: {e}")
                except Exception as e:
                    st.exception(f"Error during summarization: {e}")
