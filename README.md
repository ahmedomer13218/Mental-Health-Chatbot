
# EL_Metabtabaty: Mental Health Chatbot
EL_Metabtabaty is a mental health chatbot designed to provide support, resources, and empathetic conversations. Developed as part of a graduation project for the Samsung Innovation Campus, this chatbot uses advanced AI techniques to engage with users on mental health topics. EL_Metabtabaty operates with two core approaches: Retrieval-Augmented Generation (RAG) and Fine-Tuned Llama 3.2. The app is deployed on Streamlit, offering users an intuitive interface to access its services.

## Table of Contents
- Overview
- Features
- Data Preprocessing
- Project Structure
- Installation
- Usage
- Contributing
- License

### Overview
EL_Metabtabaty is designed to assist users by providing information, support, and empathetic responses related to mental health topics. This chatbot uses two main approaches:

- RAG Model: Integrates document retrieval and generation, allowing for contextually rich, accurate answers based on embedded documents.
- Fine-Tuned Llama 3.2: A fine-tuned conversational model, optimized with techniques like LoRA, QLoRA, and Gradient Checkpointing for high-quality and efficient dialogue generation.
dataset link : https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations

### Features
#### RAG Model (Deployed on Streamlit)
The deployed RAG model includes the following features:

- Session History: Maintains chat history across sessions for a personalized experience.
- Voice Input: Supports voice input, making interactions accessible.
- User-Friendly Interface: Streamlit-based UI that’s smooth and easy to use.
#### Fine-Tuned Llama 3.2
The alternate configuration uses a fine-tuned version of Llama 3.2, optimized with memory-efficient techniques, making it suitable for more resource-limited environments.

### Project Structure
 * [El_Metabtabaty.py](./El_Metabtabaty.py)
 * [LICENSE](./LICENSE)
 * [README.md](./README.md)
 * [data](./data)
 * [Llama3.2_Finetuning_Chatbot.ipynb](./Llama3.2_Finetuning_Chatbot.ipynb)


### Data Preprocessing:
- Data Ingestion: CSV file reading for mental health conversations
- Text Splitting: Splits conversations into manageable chunks for processing
- Embedding: Hugging Face all-MiniLM-L6-v2 embeddings for creating vector representations
- VectorStore DB: FAISS, used to store and query embeddings efficiently

### Installation
- Clone this repository:
```bash
git clone https://github.com/ahmedomer13218/Mental-Health-Chatbot

cd Chatbot_mentalHealth_directory

```

### Usage
- Run the Streamlit app to interact with the chatbot :
```bash
streamlit run EL_Metabtabaty.py
```


### Contributing
We welcome contributions! If you'd like to contribute, please fork the repository and make your changes. After testing, submit a pull request for review.

### License
This project is licensed under the MIT License - see the LICENSE file for details.




#
EL_Metabtabaty aims to be a valuable resource in mental health support by leveraging advanced AI. We are committed to making this chatbot accessible, reliable, and effective in promoting mental well-being.
