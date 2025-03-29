# 🤖 MediBot - AI-Powered Medical Chatbot

## 🏥 About MediBot
MediBot is an advanced AI-powered chatbot designed to provide **reliable medical information** using **Google Gemini AI** and **Pinecone vector search**. It retrieves context-aware answers from medical literature, ensuring accurate responses to user queries.

## 🚀 Features
- 🔍 **Retrieval-Augmented Generation (RAG):** Uses Pinecone for document retrieval.
- 🤖 **Google Gemini AI:** Generates intelligent, context-aware responses.
- 🎨 **Interactive UI:** Built with Streamlit for a seamless chat experience.
- 🔄 **Follow-Up Questions:** Maintains conversation history for better query handling.
- 🔐 **Secure API Key Management:** Uses `.env` files to protect sensitive credentials.

## 🛠️ Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure API Keys
Create a `.env` file and add:
```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 5️⃣ Run the Chatbot
```bash
streamlit run app.py
```

## 📸 Screenshots
Coming soon! 📷

## 🏗️ Future Enhancements
- 🩺 Integration with real-time medical databases.
- 🗣️ Voice-based interaction.
- 📊 More interactive analytics for user queries.

## 🤝 Contributing
Want to improve MediBot? Follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Commit changes and open a pull request.

## 📜 License
This project is licensed under the **MIT License**.

---
💡 *Disclaimer: MediBot provides medical information but is not a substitute for professional medical advice.*

