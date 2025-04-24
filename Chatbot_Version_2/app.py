import streamlit as st
import pyodbc
import ollama
from typing import Dict, Any, Optional, List
import pandas as pd
import re
from functools import lru_cache
import json
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must be the first Streamlit command
st.set_page_config(page_title="AdventureWorks Analytics", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
        color: #ffffff;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background-color: #ffffff;
        color: #000000;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        border-radius: 10px;
        padding: 15px;
        color: #000000;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        background-color: #ffffff;
        color: #000000;
    }
    .stButton>button {
        background-color: #007bff;
        color: #ffffff;
        border-radius: 5px;
        padding: 8px 15px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, small {
        color: #000000;
    }
    .sidebar h1, .sidebar h2, .sidebar h3, .sidebar p, .sidebar div, .sidebar span, .sidebar small {
        color: #ffffff !important;
    }
    .stChatInput {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 5px 10px;
        color: #000000;
    }
    .stChatInput input {
        color: #000000 !important;
    }
    .stChatInput .st-c7::placeholder {
        color: #666666;
    }
    /* Target the sidebar content */
    [data-testid="stSidebar"] {
        color: white;
    }
    /* Ensure headers and text in sidebar are white */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

class SQLChatbot:
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self.database_name = "AdventureWorks2019"
        self.schema_details = self.fetch_schema_details()
        self.system_prompt = self.load_system_prompt()
        self.chat_context = []
        self.intent_map = self.load_intent_map()
        self.init_model()

    def fetch_schema_details(self) -> str:
        """Fetch detailed schema information from the database."""
        schema_query = """
        SELECT 
            c.TABLE_SCHEMA,
            c.TABLE_NAME,
            t_desc.value AS TableDescription,
            c.COLUMN_NAME,
            c.DATA_TYPE,
            col_desc.value AS ColumnDescription,
            CASE 
                WHEN pk.COLUMN_NAME IS NOT NULL THEN 'Primary Key'
                ELSE ''
            END AS PrimaryKey,
            CASE 
                WHEN fk.COLUMN_NAME IS NOT NULL THEN 'Foreign Key to ' + fk.REFERENCED_TABLE_NAME + '(' + fk.REFERENCED_COLUMN_NAME + ')'
                ELSE ''
            END AS ForeignKey
        FROM INFORMATION_SCHEMA.COLUMNS c
        LEFT JOIN (
            SELECT 
                tc.TABLE_SCHEMA, 
                tc.TABLE_NAME, 
                kcu.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
            WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
        ) pk ON c.TABLE_SCHEMA = pk.TABLE_SCHEMA 
              AND c.TABLE_NAME = pk.TABLE_NAME 
              AND c.COLUMN_NAME = pk.COLUMN_NAME
        LEFT JOIN (
            SELECT 
                fk.TABLE_SCHEMA,
                fk.TABLE_NAME,
                cu.COLUMN_NAME,
                pk.TABLE_NAME AS REFERENCED_TABLE_NAME,
                pku.COLUMN_NAME AS REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
            JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS fk 
                ON rc.CONSTRAINT_NAME = fk.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE cu 
                ON fk.CONSTRAINT_NAME = cu.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS pk 
                ON rc.UNIQUE_CONSTRAINT_NAME = pk.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE pku 
                ON pk.CONSTRAINT_NAME = pku.CONSTRAINT_NAME
        ) fk ON c.TABLE_SCHEMA = fk.TABLE_SCHEMA 
              AND c.TABLE_NAME = fk.TABLE_NAME 
              AND c.COLUMN_NAME = fk.COLUMN_NAME
        LEFT JOIN sys.extended_properties t_desc
            ON t_desc.major_id = OBJECT_ID(c.TABLE_SCHEMA + '.' + c.TABLE_NAME)
            AND t_desc.minor_id = 0
            AND t_desc.class = 1
            AND t_desc.name = 'MS_Description'
        LEFT JOIN sys.extended_properties col_desc
            ON col_desc.major_id = OBJECT_ID(c.TABLE_SCHEMA + '.' + c.TABLE_NAME)
            AND col_desc.minor_id = (SELECT column_id FROM sys.columns WHERE name = c.COLUMN_NAME AND object_id = OBJECT_ID(c.TABLE_SCHEMA + '.' + c.TABLE_NAME))
            AND col_desc.class = 1
            AND col_desc.name = 'MS_Description'
        ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION;
        """
        try:
            with self.connect_to_database() as conn:
                df = pd.read_sql(schema_query, conn)
                schema_text = "Database Schema Details:\n"
                for _, row in df.iterrows():
                    schema_text += (
                        f"- Table: [{row['TABLE_SCHEMA']}].[{row['TABLE_NAME']}]\n"
                        f"  Description: {row['TableDescription'] or 'No description'}\n"
                        f"  Column: {row['COLUMN_NAME']} (Type: {row['DATA_TYPE']})\n"
                        f"    Description: {row['ColumnDescription'] or 'No description'}\n"
                        f"    Primary Key: {row['PrimaryKey']}\n"
                        f"    Foreign Key: {row['ForeignKey']}\n"
                    )
                logger.info("Schema details fetched successfully.")
                currency_details = df[df['TABLE_NAME'].str.lower() == 'currency'].to_string()
                logger.info(f"Currency table schema:\n{currency_details}")
                logger.info(f"Full schema details (truncated):\n{schema_text[:1000]}...")
                return schema_text
        except Exception as e:
            logger.error(f"Failed to fetch schema details: {str(e)}")
            return "Failed to load schema details. Proceeding with limited schema knowledge."

    def load_system_prompt(self):
        """Load the system prompt tailored for AdventureWorks2019 with schema details."""
        with open(r"C:\Users\acer\OneDrive\Documentos\GitHub\Chatbot\json_files\sql_queries_examples.json", 'r') as queries_file:
            example_queries = json.load(queries_file)
        
        with open(r"C:\Users\acer\OneDrive\Documentos\GitHub\Chatbot\json_files\sql_critical_rules.json", 'r') as rules_file:
            rules = json.load(rules_file)
        
        rules_text = "\n".join([f"- {rule['rule']}: {rule['description']}" for category in rules['critical_rules'] for rule in rules['critical_rules'][category]])
        examples_text = "\n".join([f"- **{table}**: ```sql\n{data['example_query']}\n```" for table, data in rules['table_examples'].items()])
        join_examples = "\n".join([f"- {example['description']}: ```sql\n{example['sql']}\n```" for example in rules['join_examples']])
        
        return f"""
        You are a SQL expert for the AdventureWorks2019 database. Study the following schema details carefully before answering any questions. Use only the columns and tables listed here, and do not invent columns that do not exist:

        {self.schema_details}

        Follow these CRITICAL RULES:

        CRITICAL RULES:
        {rules_text}
        - When querying, always include the FROM clause with schema-qualified table names
        - Database name is [AdventureWorks2019]
        - Common schemas: [HumanResources], [Sales], [Production], [Person], [Purchasing]
        - Always use appropriate table aliases (e.g., 'e' for Employee, 'soh' for SalesOrderHeader)
        - For simple "show table" requests, use SELECT TOP 100 * FROM [schema].[table] without adding unnecessary columns or conditions
        - Never prepend 'Table' to the table name; use only the exact table name as provided by the user
        - When a specific table is requested, first find the table in the schema details, then determine its schema

        **EXAMPLES OF QUERIES FOR EACH TABLE:**
        {examples_text}

        **PROPER JOIN EXAMPLES:**
        {join_examples}

        Remember:
        - Always use schema qualification (e.g., [AdventureWorks2019].[Sales].[SalesOrderHeader])
        - Fully qualify both sides of JOIN conditions
        - Use consistent table aliases
        - Return only SELECT queries, no modifications
        - Stick strictly to the schema details provided above

        Respond ONLY with the SQL query in ```sql``` blocks. No explanations.
        """

    def load_intent_map(self):
        """Define intent map for general questions."""
        return {
            "greeting": {
                    "keywords": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"], 
                    "response": self.handle_greeting
            },
            "self_intro": {
                    "keywords": ["who are you", "what are you", "tell me about yourself"], 
                    "response": lambda _: "I'm the AdventureWorks2019 Chatbot, here to help with your database queries!"
            },
            "date_query": {
                    "keywords": ["what is the date", "today's date", "current date"],
                    "response": lambda _: f"Today is {datetime.now().strftime('%B %d, %Y')}."
            },
            "well_being": {
                    "keywords": ["how are you", "are you okay", "how you doing"], 
                    "response": lambda _: "I'm doing great, thanks for asking! How can I help you with AdventureWorks2019?"
            }
        }

    def init_model(self):
        """Preload the model for faster responses."""
        try:
            ollama.pull('llama3.2')
            st.success("Model ready!", icon="✅")
        except Exception as e:
            st.error(f"Model loading failed: {e}", icon="❌")
            raise

    def handle_greeting(self, user_input: str) -> str:
        """Dynamic greeting based on time of day."""
        current_hour = datetime.now().hour
        user_input = user_input.lower().strip()
        
        if "morning" in user_input and "show" not in user_input:
            return "Good morning! How can I assist you with AdventureWorks2019 today?" if 5 <= current_hour < 12 else "Good morning to you too! It’s not morning here, but how can I help?"
        elif "afternoon" in user_input and "show" not in user_input:
            return "Good afternoon! How can I assist you with AdventureWorks2019 today?" if 12 <= current_hour < 17 else "Good afternoon to you too! It’s not afternoon here, but how can I help?"
        elif "evening" in user_input and "show" not in user_input:
            return "Good evening! How can I assist you with AdventureWorks2019 today?" if 17 <= current_hour < 22 else "Good evening to you too! It’s not evening here, but how can I help?"
        elif any(kw in user_input for kw in ["hi", "hello", "hey"]) and "show" not in user_input:
            return "Hello! How can I assist you with AdventureWorks2019 today?"
        return None

    def detect_intent(self, user_input: str) -> Optional[str]:
        """Detect intent and return response for general questions."""
        user_input_lower = user_input.lower().strip()
        for intent, config in self.intent_map.items():
            if any(keyword in user_input_lower for keyword in config["keywords"]):
                response = config["response"](user_input)
                if response:
                    return response
        return None

    def is_out_of_scope(self, user_input: str) -> bool:
        """Check if the request is out of scope."""
        return any(keyword in user_input.lower() for keyword in ["image", "video", "weather", "news", "joke", "story"])

    def is_follow_up_question(self, user_input: str) -> bool:
        """Detect if the input is a follow-up question."""
        follow_up_keywords = ["more", "details", "filter", "by", "about", "that", "last", "next", "previous"]
        return any(keyword in user_input.lower() for keyword in follow_up_keywords) and self.chat_context

    def validate_group_by(self, query: str) -> bool:
        """Validate that all non-aggregated columns in SELECT are in GROUP BY."""
        select_match = re.search(r"SELECT\s+(?:TOP\s+\d+\s+)?(.*?)\s+FROM", query, re.IGNORECASE | re.DOTALL)
        group_by_match = re.search(r"GROUP BY\s+(.*?)(?:ORDER BY|\n|;|$)", query, re.IGNORECASE | re.DOTALL)
        
        if not select_match or not group_by_match:
            return True 
        
        select_columns = [re.sub(r"\b(e|soh|p|d)\.", "", col.strip()) for col in select_match.group(1).split(",")]
        select_columns = [re.sub(r"\s+AS\s+.*", "", col) for col in select_columns]
        group_by_columns = [re.sub(r"\b(e|soh|p|d)\.", "", col.strip()) for col in group_by_match.group(1).split(",")]
        
        missing_columns = [col for col in select_columns if not re.search(r"\b(SUM|COUNT|AVG|MIN|MAX)\b", col, re.IGNORECASE) and col not in group_by_columns]
        if missing_columns:
            raise ValueError(f"Invalid GROUP BY clause: Missing columns {', '.join(missing_columns)}")
        return True

    def extract_sql(self, response: str) -> str:
        """Extract SQL from markdown code block or raw response."""
        match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
        query = match.group(1).strip() if match else response.strip()
        
        if not re.match(r"^\s*SELECT\b", query, re.IGNORECASE):
            raise ValueError("Generated query is not a SELECT statement")
        if re.search(r"\b(DELETE|UPDATE|DROP|TRUNCATE|INSERT|ALTER)\b", query, re.IGNORECASE):
            raise ValueError("Destructive operations are not allowed")
        if not re.search(r"\bFROM\b", query, re.IGNORECASE):
            raise ValueError("Generated query is missing the FROM clause")
        if not re.search(r"\[(HumanResources|Sales|Production|Person|Purchasing)\]\.", query):
            raise ValueError("Query must use AdventureWorks2019 schema qualification")
        
        self.validate_group_by(query)
        return query

    def generate_sql_query(self, user_question: str) -> str:
        """Generate SQL with context for follow-up questions."""
        user_question_lower = user_question.lower()
        if "show" in user_question_lower and "table" in user_question_lower:
            patterns = [
                r"show\s+(?:me\s+)?the\s+table\s+([\w\s]+)",
                r"show\s+(?:me\s+)?the\s+([\w\s]+)\s+table",
                r"show\s+(?:me\s+)?([\w\s]+)\s+table"
            ]
            for pattern in patterns:
                match = re.search(pattern, user_question_lower, re.IGNORECASE)
                if match:
                    table_name = match.group(1).strip().replace(" ", "").lower()
                    logger.info(f"Extracted table name: {table_name}")
                    # Find table and schema from schema details
                    table_pattern = re.compile(r"Table: \[([A-Za-z]+)\]\.\[([A-Za-z]+)\]", re.IGNORECASE)
                    for line in self.schema_details.split('\n'):
                        if "Table: " in line:
                            match_table = table_pattern.search(line)
                            if match_table:
                                schema = match_table.group(1)
                                found_table = match_table.group(2)
                                if table_name.lower() == found_table.lower():
                                    full_table_name = f"[AdventureWorks2019].[{schema}].[{found_table}]"
                                    query = f"SELECT TOP 100 * FROM {full_table_name}"
                                    logger.info(f"Matched schema: {schema}, Generated table show query: {query}")
                                    return query
                    logger.warning(f"Table '{table_name}' not found in schema details. Falling back to LLM.")
        
        # Fallback to LLM for other queries or unknown tables
        context = ""
        if self.is_follow_up_question(user_question) and self.chat_context:
            last_entries = self.chat_context[-2:]
            context = "\n".join([f"Previous question {i+1}: {entry['question']}\nPrevious query {i+1}: {entry['query']}\nPrevious results {i+1}: {json.dumps(entry['results'][:5] if entry['results'] else [])}" for i, entry in enumerate(last_entries)])

        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': f"{context}\nEnsure the SQL query uses AdventureWorks2019 schema names ([HumanResources], [Sales], [Production], [Person], [Purchasing]) and includes the FROM clause with database name [AdventureWorks2019]. Use only columns listed in the schema details.\nNew query: {user_question}"}
        ]
        
        response = ollama.chat(
            model='llama3.2',
            messages=messages,
            options={'temperature': 0.05, 'num_predict': 300, 'top_p': 0.95, 'max_tokens': 500}
        )
        
        query = self.extract_sql(response['message']['content'])
        logger.info(f"Generated SQL Query (LLM fallback): {query}")
        return query

    @st.cache_data(ttl=3600, show_spinner=False)
    def execute_query(_self, query: str) -> Optional[list]:
        try:
            logger.info(f"Executing query: {query}")
            with _self.connect_to_database() as conn:
                df = pd.read_sql(query, conn)
                return df.to_dict('records')
        except pyodbc.Error as e:
            logger.error(f"SQL Execution Error: {str(e)}")
            raise ValueError(f"Execution failed on sql '{query}': {str(e)}")

    @lru_cache(maxsize=128)
    def connect_to_database(self) -> pyodbc.Connection:
        """Cached database connection for AdventureWorks2019."""
        conn_str = (
            f"DRIVER={{{self.connection_params['driver']}}};"
            f"SERVER={self.connection_params['server']};"
            f"DATABASE={self.connection_params['database']};"
            f"Trusted_Connection={self.connection_params['trusted_connection']};"
        )
        return pyodbc.connect(conn_str)

    def generate_natural_language_response(self, user_question: str, query: str, query_results: List[Dict[str, Any]]) -> str:
        """Generate a natural language response based on AdventureWorks2019 query results."""
        formatted_results = '\n'.join([f"- {', '.join([f'{k}: {v}' for k, v in row.items()])}" for row in query_results[:10]])
        context = ""
        if self.is_follow_up_question(user_question) and self.chat_context:
            last_entries = self.chat_context[-2:]
            context = "\n".join([f"Previous question {i+1}: {entry['question']}\nPrevious results {i+1}: {json.dumps(entry['results'][:5] if entry['results'] else [])}" for i, entry in enumerate(last_entries)])
        
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant for AdventureWorks2019 database queries.\n- Use the executed SQL query to understand the data structure and intent.\n- Focus on answering the user's question using the results.\n- If the query uses TOP or ORDER BY, highlight the ranking/selection in the response.\n- Tailor the answer to the specific AdventureWorks2019 data.\n- Maintain context from previous questions if applicable."},
            {'role': 'user', 'content': f"{context}\nQuestion: {user_question}\nExecuted SQL Query: {query}\nQuery Results: \n{formatted_results}\nProvide a clear, concise answer based on the question, query, and results."}
        ]
        response = ollama.chat(model='llama3.2', messages=messages, options={'temperature': 0.03, 'num_predict': 300, 'top_p': 0.95})
        return response['message']['content']

    def chat(self, user_input: str, show_results: bool = False) -> Dict[str, Any]:
        """Handle chat interaction with option to show results."""
        logger.info(f"Processing user input: {user_input}")
        
        general_response = self.detect_intent(user_input)
        if general_response:
            response_dict = {"question": user_input, "response": general_response, "results": None, "query": None}
            self.chat_context.append(response_dict)
            return response_dict

        if self.is_out_of_scope(user_input):
            response_dict = {"question": user_input, "response": "Sorry, I can only assist with AdventureWorks2019 database queries.", "results": None, "query": None}
            self.chat_context.append(response_dict)
            return response_dict

        try:
            user_input_lower = user_input.lower()
            is_show_table_request = "show" in user_input_lower and "table" in user_input_lower
            
            query = self.generate_sql_query(user_input)
            results = self.execute_query(query) if query else None
            
            if not results:
                response_dict = {"question": user_input, "query": query, "response": "No data found for this query in AdventureWorks2019.", "results": None}
                self.chat_context.append(response_dict)
                return response_dict

            if is_show_table_request:
                response_dict = {"question": user_input, "query": query, "response": f"Results from AdventureWorks2019 query: ```sql\n{query}\n```", "results": results}
                self.chat_context.append(response_dict)
                return response_dict

            natural_language_response = self.generate_natural_language_response(user_input, query, results)
            response_dict = {"question": user_input, "query": query, "response": natural_language_response, "results": results if show_results else None}
            self.chat_context.append(response_dict)
            return response_dict
        except Exception as e:
            response_dict = {"question": user_input, "response": f"Sorry, I couldn't process this request: {str(e)}", "results": None, "query": None}
            self.chat_context.append(response_dict)
            return response_dict

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None

def display_results(results):
    if not results:
        st.warning("No results found", icon="⚠️")
        return
    
    df = pd.DataFrame(results)
    st.session_state.current_df = df
    st.dataframe(df, use_container_width=True, height=min(400, 35 * (len(df) + 1)), hide_index=False)

def main():
    initialize_session_state()

    # Database configuration for AdventureWorks2019
    db_config = {
        'driver': "ODBC Driver 17 for SQL Server",
        'server': "LAPTOP-556912EV",
        'database': "AdventureWorks2019",
        'trusted_connection': "yes"
    }

    # Sidebar
    with st.sidebar:
        st.header("AdventureWorks Chatbot")
        st.markdown("**Powered by llama3.2 LLM**")
        st.markdown("---")
        st.subheader("Options")
        show_sql = st.checkbox("Show SQL Queries", value=False)
        clear_chat = st.button("Clear Chat History")
        if clear_chat:
            st.session_state.chat_history = []
            st.rerun()
        st.markdown("---")
        st.info("Connected to AdventureWorks2019", icon="✅")

    # Main content
    st.title("📊 AdventureWorks Analytics")
    st.markdown(f"**Welcome!** Ask anything about your AdventureWorks2019 data. *Updated: {datetime.now().strftime('%B %d, %Y')}*")

    col1, col2 = st.columns([2, 1])
    with col1:
        # Chat container
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                with st.chat_message("ai", avatar="🤖"):
                    st.markdown("Hello! I'm here to help with AdventureWorks2019 queries.")
                    st.caption("Try: 'Show me the table Currency' or 'What are the top 10 sales orders?'")

            for message in st.session_state.chat_history:
                avatar = "👤" if message["role"] == "user" else "🤖"
                with st.chat_message(message["role"], avatar=avatar):
                    if message["role"] == "user":
                        st.markdown(f"**Question:** {message['content']}")
                    else:
                        st.markdown(message["content"])
                        if show_sql and message.get("query"):
                            st.code(message["query"], language="sql")
                        if message.get("results"):
                            display_results(message["results"])
                    st.markdown(f"<small>{message['timestamp']}</small>", unsafe_allow_html=True)

    with col2:
        st.subheader("Quick Tips")
        st.markdown("""
        - Use "show table [name]" to view data (e.g., Currency, Employee)
        - Ask about sales, employees, products, etc.
        - Try "top 10" or "by date" queries
        - Add "show results" for detailed output
        """)

    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**Question:** {prompt}<br><small>{datetime.now().strftime('%H:%M:%S')}</small>", unsafe_allow_html=True)
        
        with st.spinner("Analyzing your query..."):
            show_results = any(phrase in prompt.lower() for phrase in ["show results", "display results"])
            st.session_state.chat_history.append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%H:%M:%S")})
            if 'chatbot' not in st.session_state:
                st.session_state.chatbot = SQLChatbot(db_config)
            response = st.session_state.chatbot.chat(prompt, show_results)
            st.session_state.chat_history.append({
                "role": "ai",
                "content": response["response"],
                "results": response.get("results"),
                "query": response.get("query"),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()

if __name__ == "__main__":
    main()