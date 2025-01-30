# import required libraries
import streamlit as st
import pyodbc
import ollama
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
import re
from functools import lru_cache

class SQLChatbot:
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self.system_prompt = """
        You are a SQL expert for AdventureWorks2019. Follow these rules:

        1. Use fully qualified names (e.g., AdventureWorks2019.Sales.SalesOrderHeader)
        2. Always use table aliases
        3. Fully qualify JOIN conditions
        4. Use TOP 100 unless user specifies otherwise
        5. Return ONLY SQL wrapped in a ```sql code block. No explanations.

        Example:
        ```sql
        SELECT TOP 10 p.ProductID, p.Name, SUM(sod.LineTotal) AS TotalSales
        FROM AdventureWorks2019.Sales.SalesOrderDetail AS sod
        JOIN AdventureWorks2019.Production.Product AS p ON p.ProductID = sod.ProductID
        GROUP BY p.ProductID, p.Name
        ORDER BY TotalSales DESC;
        ```

        Respond ONLY with the SQL query in ```sql. No explanations.
        """
        self.init_model()

    def init_model(self):
        """Preload the model for faster responses"""
        try:
            ollama.pull('codellama:7b')
            st.success("Model ready!")
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            raise

    def extract_sql(self, response: str) -> str:
        """Extract SQL from markdown code block or raw response"""
        # Try to extract from code block
        match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
        if match:
            query = match.group(1).strip()
        else:
            # Fallback: Use the entire response as the query
            query = response.strip()
        
        # Validate the query
        if not re.match(r"^\s*SELECT", query, re.IGNORECASE):
            raise ValueError("Generated query is not a SELECT statement")
        return query

    def generate_sql_query(self, user_question: str, previous_error: str = None) -> str:
        """Generate SQL with error correction support"""
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': f"{previous_error}\nNew query: {user_question}"} 
                if previous_error else
                {'role': 'user', 'content': user_question}
        ]
        
        response = ollama.chat(
            model='codellama:7b',
            messages=messages,
            options={'temperature': 0.1, 'num_predict': 200}
        )
        
        return self.extract_sql(response['message']['content'])

    @st.cache_data(ttl=3600, show_spinner=False)
    def execute_query(_self, query: str) -> Optional[list]:
        """Cached query execution with pandas"""
        try:
            with _self.connect_to_database() as conn:
                df = pd.read_sql(query, conn)
                return df.to_dict('records')
        except pyodbc.Error as e:
            raise ValueError(f"SQL Error: {str(e)}")

    @lru_cache(maxsize=128)
    def connect_to_database(self) -> pyodbc.Connection:
        """Cached database connection"""
        conn_str = (
            f"DRIVER={{{self.connection_params['driver']}}};"
            f"SERVER={self.connection_params['server']};"
            f"DATABASE={self.connection_params['database']};"
            f"Trusted_Connection={self.connection_params['trusted_connection']};"
        )
        return pyodbc.connect(conn_str)

    def chat(self, user_input: str, execute: bool = True) -> Dict[str, Any]:
        """Enhanced chat with error correction"""
        max_retries = 2
        query, error = None, None
        
        for attempt in range(max_retries + 1):
            try:
                query = self.generate_sql_query(user_input, error)
                if not execute:
                    return {
                        "question": user_input,
                        "query": query,
                        "results": None
                    }
                results = self.execute_query(query)
                return {
                    "question": user_input,
                    "query": query,
                    "results": results
                }
            except Exception as e:
                if attempt < max_retries:
                    error = f"Previous error: {str(e)}"
                    continue
                return {"error": str(e), "query": query}

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None

def display_results(results):
    """Optimized results display"""
    if not results:
        return st.warning("No results found")
    
    df = pd.DataFrame(results)
    st.session_state.current_df = df
    st.dataframe(
        df,
        use_container_width=True,
        height=min(400, 35 * (len(df) + 1))
    )

def main():
    st.set_page_config(
        page_title="AW Query Assistant",
        page_icon="🚀",
        layout="centered"
    )
    initialize_session_state()
    
    st.title("🚀 AdventureWorks Analytics")
    st.caption("Powered by CodeLlama 7B | Response time <15s typical")
    
    # Database configuration
    db_config = {
        'driver': "ODBC Driver 17 for SQL Server",
        'server': "LAPTOP-556912EV",
        'database': "AdventureWorks2019",
        'trusted_connection': "yes"
    }
    
    if not st.session_state.chat_history:
        with st.chat_message("ai"):
            st.markdown("Ask me anything about sales, products, or employees!")
            st.caption("Try: 'Top 10 products by revenue last quarter'")
    
    # Chat interface
    if prompt := st.chat_input("Ask your question..."):
        with st.spinner("🧠 Generating SQL..."):
            chatbot = SQLChatbot(db_config)
            
            with st.chat_message("user"):
                st.write(prompt)
            
            response = chatbot.chat(prompt)
            
            with st.chat_message("ai"):
                if "error" in response:
                    st.error(f"Failed: {response['error']}")
                    if response.get('query'):
                        st.code(response['query'], language='sql')
                else:
                    st.code(response['query'], language='sql')
                    display_results(response['results'])
                    st.session_state.chat_history.append({
                        'question': prompt,
                        'query': response['query'],
                        'results': response['results'],
                        'time': datetime.now().strftime("%H:%M:%S")
                    })

if __name__ == "__main__":
    main()