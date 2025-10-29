# INFO 5940 
Welcome to the INFO 5940 repository. You will complete your work using [**GitHub Codespaces**](#about-github-codespaces) and save your progress in your own GitHub repository. This guide will walk you through setting up the development environment and running the test notebook.  

## Instructions to run

1. Make sure you are on the assignment1 branch
2. Install necessary packages with `pip install -r requirements.txt`
2. Add your API Tokens to your enviornment using `export API_KEY="token_here"` and `export OPENAI_API_KEY="token_here"`
3. Start the streamlit app with `streamlit run chat_with_pdf.py`

## Implementation details

- Uploaded files are read and split into text chunks
- Chunks are then embedded using `text-embeding-3-large`
- These are stored locally using ChromaDB
- The app retrieves the msot relevant chunks using similarity search and uses gpt-4o-mini to answer based only on those chunks
- The app also is prompted to cite sources so the user knows where the info came from

## Changes to configuration

- There were no changes to the existing configuration other than adding `chromadb` to the `requirments.txt` file.