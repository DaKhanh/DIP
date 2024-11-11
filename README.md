# DIP

## Technology Stack
Retrieval-Augmented Generation (RAG), FAISS (Facebook AI Similarity Search), Flask.

## How to run?

1. **Download the Project**: Either download as zip and extract the project files or clone the repository using Git.

2. **Set Up the Environment**: Open a terminal, navigate to the project directory, and run the following commands to set up the environment:

   ```bash
   conda env create -f environment.yml
   conda activate dip_all
   ```

   



3. **Add OpenAI Key**:In the `.env` file, write your OpenAI API key like this:

   ```bash
   OPENAI_API_KEY="your-openai-key"
   ```

   

4. **Start the Application**:

   ```bash
   python run.py
   ```

   

5. **Access the Application**: Open any HTML page with browser, and start interacting by asking questions.
