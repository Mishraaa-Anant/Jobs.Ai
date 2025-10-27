# Jobs.Ai
This Python script implements an advanced Applicant Tracking System (ATS) designed to streamline the initial stages of recruitment. Built with Streamlit, it provides an interactive web interface for recruiters to analyze candidate resumes against job descriptions.

# 🎯 Next-Gen ATS Resume Shortlister Pro

This project is an AI-powered Applicant Tracking System (ATS) built with Python and Streamlit. It uses semantic search to find relevant resumes and a local Large Language Model (LLM) via Ollama for deep analysis and scoring against a job description. It also includes features for automated candidate outreach via SendGrid.

## ✨ Features

* **Semantic Resume Matching:** Uses `sentence-transformers` to find resumes that contextually match the job description, going beyond simple keyword search.
* **AI-Powered ATS Analysis:** Leverages a local LLM (e.g., Llama 3.2 via Ollama) to perform detailed analysis of shortlisted resumes, providing scores for skills, experience, education, and keyword match.
* **Structured Job Description Deconstruction:** Uses the LLM to understand the core requirements (must-haves vs. nice-to-haves) of the job description *before* analyzing resumes.
* **Generative Summaries & Ranking:** The LLM generates concise summaries for each candidate and provides a final comparative ranking of the top candidates.
* **Hybrid Contact Extraction:** Robustly extracts candidate name, email, phone, and LinkedIn using a combination of RegEx and LLM fallback. Handles common email obfuscations.
* **Automated Email Invitations:** Integrates with SendGrid to send interview invitations to selected candidates, automatically generating unique Jitsi meeting links.
* **Web Interface:** Easy-to-use web UI built with Streamlit for configuration, analysis, and viewing results.
* **Caching:** Implements multiple layers of caching (model, embeddings, ATS scores) for faster performance.
* **Error Handling & Debugging:** Includes robust error handling for LLM calls, JSON parsing, and provides debugging information in the UI for failed analyses.

## 🛠️ Technology Stack

* **Backend:** Python 3.9+
* **Web Framework:** Streamlit
* **AI/ML:**
    * `sentence-transformers` (for embeddings & semantic search)
    * Ollama (to run local LLMs like Llama 3.2, Mistral, etc.)
    * PyTorch (dependency for sentence-transformers)
* **PDF Processing:** `PyPDF2`
* **Email:** SendGrid API
* **Data Handling:** Pandas, NumPy
* **Configuration:** `python-dotenv`
* **API Calls:** `requests`

## 🚀 Setup and Installation

### Prerequisites

1.  **Python:** Ensure you have Python 3.9 or later installed.
2.  **Ollama:** Install and run Ollama locally. Pull the model you want to use (default is `llama3.2`):
    ```bash
    ollama pull llama3.2
    ```
    Make sure Ollama is serving the model (usually by just running `ollama serve` in a separate terminal if it's not running as a service).
3.  **SendGrid Account:** Sign up for a SendGrid account and create an API key with Mail Send permissions.

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file containing all the imported libraries: `streamlit`, `torch`, `numpy`, `pandas`, `python-dotenv`, `pypdf2`, `sentence-transformers`, `sendgrid`, `requests`)*

4.  **Configure Environment Variables:**
    Create a file named `.env` in the project root directory and add the following, replacing the placeholder values:
    ```env
    # SendGrid Configuration
    SENDGRID_API_KEY="YOUR_SENDGRID_API_KEY"
    SENDER_EMAIL="your_verified_sender@example.com"

    # Ollama Configuration (Optional - Defaults are shown)
    OLLAMA_API_URL="http://localhost:11434"
    OLLAMA_MODEL="llama3.2" # Or whichever model you pulled

    # File Paths (Optional - Defaults are shown)
    RESUME_FOLDER="resumes"
    EMBEDDING_CACHE="resume_embeddings.pkl"
    ```

5.  **Create the Resume Folder:** 📁
    **Manually create a folder named `resumes`** in the project's root directory (or use the name you specified for `RESUME_FOLDER` in your `.env` file).

6.  **Add Resumes:**
    Place candidate resume **PDF files** into the `resumes` folder you just created.

### Running the Application

1.  Ensure your Ollama server is running and serving the specified model.
2.  Run the Streamlit app from your terminal:
    ```bash
    streamlit run app.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## 📋 Usage

1.  **Configure:** Use the sidebar to:
    * Verify Ollama connection.
    * Enter the **Job Title** and paste the full **Job Description**.
    * Adjust **Filters** like the Minimum ATS Score required and the Maximum Resumes to analyze deeply.
2.  **Analyze:** Click the **"Start ATS Analysis"** button.
    * The app will first deconstruct the JD.
    * Then, it performs semantic search to find the most relevant resumes.
    * Finally, it sends the top resumes to Ollama for detailed scoring. This step might take some time depending on the number of resumes and your hardware.
3.  **Review Results:**
    * A **Manager's Ranking** of the top candidates will appear first.
    * Use the **"Candidate List"** tab to see all qualified candidates, ranked by score, with AI summaries and key details. Expand entries for more info.
    * Use the **"Deep Analysis"** tab to select a specific candidate and view their full ATS breakdown, including skill matches, experience/education validation, strengths, red flags, and the raw JSON output.
    * If LLM analysis failed for a candidate, a debug expander will appear in the main analysis section showing the raw LLM response.
4.  **Send Invitations:**
    * Go to the **"Send Invitations"** tab.
    * Select the candidates you want to invite (only those with detected emails are shown).
    * Click **"Send Interview Invitations"**. Emails will be sent via SendGrid with unique Jitsi links. The status will update next to each candidate.

---
