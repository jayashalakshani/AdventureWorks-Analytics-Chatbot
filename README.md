# 🚀 AdventureWorks Analytics Chatbot

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-00A98F?style=for-the-badge&logo=Ollama&logoColor=white)
![SQL Server](https://img.shields.io/badge/SQL_Server-CC2927?style=for-the-badge&logo=microsoft-sql-server&logoColor=white)

An intelligent chatbot powered by **CodeLlama 7B** and **Streamlit** that generates SQL queries for the **AdventureWorks2019** database. Simply ask questions in natural language, and the chatbot will generate and execute the corresponding SQL queries.

---

## **Features**

- **Natural Language to SQL**: Converts user questions into SQL queries using Ollama's CodeLlama 7B model.
- **Interactive Interface**: Built with Streamlit for a seamless user experience.
- **Cached Queries**: Results are cached for faster repeated queries.
- **Error Handling**: Retries query generation up to 2 times with improved error context.
- **Fully Qualified SQL**: Ensures all queries use fully qualified table names and aliases.

---

## **How It Works**

1. **User Input**: Ask a question in natural language, e.g., "Show me top 10 products by sales."
2. **SQL Generation**: The chatbot uses CodeLlama 7B to generate a SQL query.
3. **Query Execution**: The query is executed against the AdventureWorks2019 database.
4. **Results Display**: The results are displayed in an interactive dataframe.
