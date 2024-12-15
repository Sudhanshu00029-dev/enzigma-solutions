import streamlit as st
import json
import pandas as pd
from datetime import datetime
import re
import uuid
import sqlite3
import os
import time
from typing import Dict, List
import google.generativeai as genai
from collections import deque
import threading
import io
import zipfile

class RateLimiter:
    def __init__(self, max_requests=60, time_frame=60):
        """
        Initialize RateLimiter with configurable max requests and time frame
        
        :param max_requests: Maximum number of requests allowed in the time frame
        :param time_frame: Time frame in seconds
        """
        self.max_requests = max_requests
        self.time_frame = time_frame
        self.request_times = deque()
        self._lock = threading.Lock()

    def wait(self):
        """
        Wait if request rate exceeds the limit
        Implements thread-safe rate limiting logic
        """
        current_time = time.time()
        
        with self._lock:
            # Remove timestamps outside the current time frame
            while self.request_times and current_time - self.request_times[0] > self.time_frame:
                self.request_times.popleft()
            
            # Check if we've exceeded max requests
            if len(self.request_times) >= self.max_requests:
                # Calculate wait time
                oldest_request = self.request_times[0]
                wait_time = self.time_frame - (current_time - oldest_request)
                
                if wait_time > 0:
                    st.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
            
            # Add current request time
            self.request_times.append(current_time)

class SchemaGenerator:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=st.session_state.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.rate_limiter = RateLimiter()

    def generate_schema(self, prompt):
        """
        Generate schema with rate limiting
        
        :param prompt: User's schema description
        :return: Generated schema dictionary
        """
        try:
            # Apply rate limiting
            self.rate_limiter.wait()
            
            # Generate schema using Gemini
            response = self.model.generate_content(
                f"Generate a JSON schema based on this description: {prompt}. "
                "Ensure the schema is valid JSON and includes appropriate types and constraints."
            )
            
            # Extract and parse JSON from response
            schema_text = response.text
            
            # Extract JSON from code block or text
            json_match = re.search(r'```json\n(.*?)```', schema_text, re.DOTALL)
            if json_match:
                schema_text = json_match.group(1)
            
            # Parse the schema
            schema = json.loads(schema_text)
            return schema
        
        except genai.types.generation_types.BlockedPromptException as e:
            st.error(f"Prompt blocked: {e}")
            return None
        except json.JSONDecodeError:
            st.error("Could not parse generated schema. Please try again.")
            return None
        except Exception as e:
            st.error(f"Error generating schema: {str(e)}")
            return None

def init_db():
    """Initialize SQLite database for storing schemas"""
    conn = sqlite3.connect('schema_records.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS schemas (
            id TEXT PRIMARY KEY,
            prompt TEXT,
            schema TEXT,
            created_at DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def check_api_key_validity():
    """
    Validate Gemini API key
    
    :return: Boolean indicating API key validity
    """
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        test_model = genai.GenerativeModel('gemini-pro')
        test_model.generate_content("Test")
        return True
    except Exception:
        return False

class SchemaGeneratorUI:
    def __init__(self):
        # Initialize database
        init_db()
        
        # Initialize schema generator
        self.schema_generator = None

    def _upload_schema_files(self, uploaded_files):
        """Handle multiple schema file uploads"""
        successful_uploads = []
        failed_uploads = []
        
        for uploaded_file in uploaded_files:
            try:
                # Read the uploaded file
                file_content = uploaded_file.getvalue().decode('utf-8')
                
                # Try parsing as JSON
                schema = json.loads(file_content)
                
                # Save to database
                conn = sqlite3.connect('schema_records.db')
                c = conn.cursor()
                schema_id = str(uuid.uuid4())
                c.execute(
                    "INSERT INTO schemas (id, prompt, schema, created_at) VALUES (?, ?, ?, ?)",
                    (schema_id, f"Uploaded: {uploaded_file.name}", json.dumps(schema), datetime.now())
                )
                conn.commit()
                conn.close()
                
                successful_uploads.append({
                    'filename': uploaded_file.name,
                    'id': schema_id,
                    'schema': schema
                })
            except json.JSONDecodeError:
                failed_uploads.append({
                    'filename': uploaded_file.name,
                    'error': "Invalid JSON format"
                })
            except Exception as e:
                failed_uploads.append({
                    'filename': uploaded_file.name,
                    'error': str(e)
                })
        
        # Provide feedback
        if successful_uploads:
            st.success(f"{len(successful_uploads)} file(s) uploaded successfully!")
            
            # Expander for successful uploads
            with st.expander("Successful Uploads"):
                for upload in successful_uploads:
                    st.subheader(f"File: {upload['filename']} (ID: {upload['id']})")
                    st.json(upload['schema'])
        
        if failed_uploads:
            st.warning(f"{len(failed_uploads)} file(s) failed to upload")
            
            # Expander for failed uploads
            with st.expander("Failed Uploads"):
                for upload in failed_uploads:
                    st.error(f"File: {upload['filename']} - Error: {upload['error']}")
        
        return successful_uploads, failed_uploads

    def _view_records(self):
        """View previously generated or uploaded schemas"""
        conn = sqlite3.connect('schema_records.db')
        c = conn.cursor()
        
        # Fetch all schemas
        c.execute("SELECT id, prompt, created_at FROM schemas ORDER BY created_at DESC")
        records = c.fetchall()
        
        if not records:
            st.info("No schema records found.")
            conn.close()
            return
        
        # Create a DataFrame for display
        df = pd.DataFrame(records, columns=['ID', 'Prompt/Source', 'Created At'])
        
        # Multiselect for bulk actions
        st.subheader("Schema Records")
        selected_ids = st.multiselect("Select schemas", df['ID'])
        
        # Display records in a table
        displayed_df = st.dataframe(
            df, 
            column_config={
                "ID": st.column_config.TextColumn("Schema ID"),
                "Prompt/Source": st.column_config.TextColumn("Description"),
                "Created At": st.column_config.TextColumn("Created")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Bulk download option
        if selected_ids:
            # Prepare a zip file of selected schemas
            schemas_to_download = {}
            for schema_id in selected_ids:
                c.execute("SELECT schema, prompt FROM schemas WHERE id = ?", (schema_id,))
                record = c.fetchone()
                if record:
                    schema = json.loads(record[0])
                    prompt = record[1]
                    schemas_to_download[f"{schema_id}_{prompt}.json"] = json.dumps(schema, indent=2)
            
            # Create zip file for download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for filename, content in schemas_to_download.items():
                    zip_file.writestr(filename, content)
            
            zip_buffer.seek(0)
            st.download_button(
                label="Download Selected Schemas",
                data=zip_buffer,
                file_name="selected_schemas.zip",
                mime="application/zip"
            )
        
        # Individual record details
        st.subheader("Record Details")
        record_id = st.text_input("Enter Schema ID to view details:")
        
        if record_id:
            c.execute("SELECT schema FROM schemas WHERE id = ?", (record_id,))
            record = c.fetchone()
            
            if record:
                schema = json.loads(record[0])
                st.json(schema)
                
                # Download button for specific record
                st.download_button(
                    label="Download Schema",
                    data=json.dumps(schema, indent=2),
                    file_name=f"schema_{record_id}.json",
                    mime="application/json"
                )
            else:
                st.error("No record found with the given ID.")
        
        conn.close()

    def run(self):
        st.title("Schema Generator")
        
        # Sidebar for API key input
        with st.sidebar:
            st.title("Configuration")
            
            # Initialize Gemini API key in session state if not exists
            if 'gemini_api_key' not in st.session_state:
                st.session_state.gemini_api_key = ''
            
            api_key = st.text_input("Enter your Gemini API key", 
                                    value=st.session_state.gemini_api_key,
                                    type="password")
            
            if api_key:
                st.session_state.gemini_api_key = api_key
                
                try:
                    # Try to initialize schema generator
                    self.schema_generator = SchemaGenerator()
                except Exception as e:
                    st.error(f"Error initializing Schema Generator: {e}")
        
        # Main interface with tabs
        tab1, tab2, tab3 = st.tabs(["Generate Schema", "Upload Schemas", "View Records"])
        
        with tab1:
            if st.session_state.gemini_api_key:
                prompt = st.text_area("Describe your schema requirements:", 
                                    height=150,
                                    help="Describe the fields and requirements for your schema.")
                
                if st.button("Generate Schema"):
                    if prompt:
                        try:
                            with st.spinner("Generating schema..."):
                                schema = self.schema_generator.generate_schema(prompt)
                                
                                if schema:
                                    # Save to database
                                    conn = sqlite3.connect('schema_records.db')
                                    c = conn.cursor()
                                    schema_id = str(uuid.uuid4())
                                    c.execute(
                                        "INSERT INTO schemas (id, prompt, schema, created_at) VALUES (?, ?, ?, ?)",
                                        (schema_id, prompt, json.dumps(schema), datetime.now())
                                    )
                                    conn.commit()
                                    conn.close()
                                    
                                    # Display results
                                    st.json(schema)
                                    
                                    # Download button
                                    st.download_button(
                                        label="Download Schema",
                                        data=json.dumps(schema, indent=2),
                                        file_name="schema.json",
                                        mime="application/json"
                                    )
                        except Exception as e:
                            st.error(f"Error generating schema: {str(e)}")
                    else:
                        st.warning("Please enter a prompt to generate the schema.")
            else:
                st.warning("Please enter your Gemini API key in the sidebar to get started.")
        
        with tab2:
            st.subheader("Upload Schema Files")
            uploaded_files = st.file_uploader(
                "Choose JSON schema files", 
                type=['json'], 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                # Perform multiple file upload
                successful_uploads, failed_uploads = self._upload_schema_files(uploaded_files)
        
        with tab3:
            self._view_records()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Schema Generator",
        page_icon="🎯",
        layout="wide"
    )
    
    # Run the application
    app = SchemaGeneratorUI()
    app.run()