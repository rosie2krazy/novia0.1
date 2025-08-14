# Enhanced Novia - AI Virtual Accountant with Advanced Document Processing
# Flask backend application with AI-powered document extraction and QuickBooks integration

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
import pandas as pd
import json
import requests
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass
import threading
from werkzeug.utils import secure_filename
import uuid
import time
import base64
from urllib.parse import urlencode, parse_qs
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import pytesseract  # OCR for images
import openpyxl
from pathlib import Path
import cv2
import numpy as np
import re
import zipfile

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configuration
@dataclass
class Config:
    QUICKBOOKS_BASE_URL = "https://sandbox-quickbooks.api.intuit.com"
    QUICKBOOKS_DISCOVERY_DOCUMENT_URL = "https://appcenter.intuit.com/api/v1/OpenID_sandbox"
    QUICKBOOKS_TOKEN_URL = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
    QUICKBOOKS_AUTH_URL = "https://appcenter.intuit.com/connect/oauth2"
    QUICKBOOKS_ONLINE_URL = "https://qbo.intuit.com"  # Added for redirection
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL_NAME = "llama3-70b-8192"
    
    # OAuth2 Configuration
    QUICKBOOKS_CLIENT_ID = os.getenv("QUICKBOOKS_CLIENT_ID", "just")
    QUICKBOOKS_CLIENT_SECRET = os.getenv("QUICKBOOKS_CLIENT_SECRET", "")
    QUICKBOOKS_REDIRECT_URI = os.getenv("QUICKBOOKS_REDIRECT_URI", "http://localhost:5000/callback")
    QUICKBOOKS_SCOPE = "com.intuit.quickbooks.accounting"
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Supported file types
    ALLOWED_EXTENSIONS = {
        'xlsx', 'xls', 'csv',  # Spreadsheets
        'pdf',  # PDFs
        'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'  # Images
    }

class DocumentProcessor:
    """Advanced document processing with AI extraction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_ocr()
    
    def setup_ocr(self):
        """Setup OCR configurations"""
        # Configure tesseract if available
        try:
            pytesseract.get_tesseract_version()
            self.ocr_available = True
        except:
            print("⚠️  Warning: Tesseract OCR not found. Image text extraction will be limited.")
            self.ocr_available = False
    
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file type is supported"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.config.ALLOWED_EXTENSIONS
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text()
                text += "\n\n"  # Separate pages
            
            doc.close()
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from images using OCR"""
        if not self.ocr_available:
            raise Exception("OCR is not available. Please install Tesseract.")
        
        try:
            # Load and preprocess image
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply image enhancements for better OCR
            # Remove noise
            denoised = cv2.medianBlur(gray, 3)
            
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Use PIL for OCR
            pil_image = Image.fromarray(enhanced)
            
            # Configure OCR settings
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()$%-@: '
            
            text = pytesseract.image_to_string(pil_image, config=custom_config)
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Error extracting text from image: {str(e)}")
    
    def extract_data_from_spreadsheet(self, file_path: str) -> Dict:
        """Extract data from spreadsheet files"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
                
                # If multiple sheets, combine them or select the first non-empty one
                if isinstance(df, dict):
                    # Find the sheet with the most data
                    best_sheet = max(df.keys(), key=lambda x: len(df[x]))
                    df = df[best_sheet]
            
            # Convert to dictionary format for easier processing
            data = {
                'columns': df.columns.tolist(),
                'rows': df.values.tolist(),
                'shape': df.shape,
                'summary': df.describe(include='all').to_dict() if len(df) > 0 else {}
            }
            
            return data
            
        except Exception as e:
            raise Exception(f"Error processing spreadsheet: {str(e)}")
    
    def ai_extract_structured_data(self, text: str, data_type: str = "auto") -> List[Dict]:
        """Use AI to extract structured data from text"""
        if not self.config.GROQ_API_KEY or self.config.GROQ_API_KEY in ["your_groq_api_key_here", ""]:
            raise Exception("Groq API key is required for AI extraction")
        
        # Create extraction prompt based on data type
        if data_type == "accounts" or "account" in text.lower():
            prompt = self._get_accounts_extraction_prompt(text)
        elif data_type == "customers" or any(word in text.lower() for word in ["customer", "client", "contact"]):
            prompt = self._get_customers_extraction_prompt(text)
        elif data_type == "invoices" or any(word in text.lower() for word in ["invoice", "bill", "receipt", "payment"]):
            prompt = self._get_invoices_extraction_prompt(text)
        else:
            prompt = self._get_auto_extraction_prompt(text)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.config.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are an expert at extracting structured financial data from documents. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000,
                "temperature": 0.1
            }
            
            response = requests.post(
                self.config.GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result["choices"][0]["message"]["content"]
                
                # Try to parse JSON response
                try:
                    # Clean the response - remove markdown formatting if present
                    cleaned_response = ai_response.strip()
                    if cleaned_response.startswith("```json"):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3]
                    
                    extracted_data = json.loads(cleaned_response)
                    
                    # Ensure we return a list
                    if isinstance(extracted_data, dict):
                        if 'data' in extracted_data:
                            return extracted_data['data']
                        else:
                            return [extracted_data]
                    elif isinstance(extracted_data, list):
                        return extracted_data
                    else:
                        return []
                        
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    print(f"AI response: {ai_response}")
                    raise Exception(f"AI returned invalid JSON: {str(e)}")
            else:
                raise Exception(f"AI API error {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"AI extraction failed: {str(e)}")
    
    def _get_accounts_extraction_prompt(self, text: str) -> str:
        return f"""
Extract chart of accounts information from the following text. Return a JSON array of account objects.

Each account object should have:
- name: Account name
- account_type: One of "Asset", "Liability", "Equity", "Income", "Expense"
- account_subtype: Appropriate subtype for QuickBooks (e.g., "Bank", "AccountsReceivable", "Expense")
- description: Brief description
- opening_balance: Initial balance if mentioned (number or null)
- opening_balance_date: Date in YYYY-MM-DD format if mentioned
- active: true/false

Text to analyze:
{text[:4000]}

Return only valid JSON array format.
"""
    
    def _get_customers_extraction_prompt(self, text: str) -> str:
        return f"""
Extract customer/client information from the following text. Return a JSON array of customer objects.

Each customer object should have:
- name: Customer full name
- company: Company name if mentioned
- email: Email address if found
- phone: Phone number if found
- address_line1: Street address
- city: City name
- postal_code: ZIP/postal code
- country: Country (default "USA")

Text to analyze:
{text[:4000]}

Return only valid JSON array format.
"""
    
    def _get_invoices_extraction_prompt(self, text: str) -> str:
        return f"""
Extract invoice information from the following text. Return a JSON array of invoice objects.

Each invoice object should have:
- customer_name: Customer name
- invoice_number: Invoice number if found
- date: Invoice date in YYYY-MM-DD format
- due_date: Due date in YYYY-MM-DD format
- items: Array of line items with description, quantity, unit_price, amount
- total_amount: Total invoice amount
- status: "Draft", "Sent", "Paid", etc.

Text to analyze:
{text[:4000]}

Return only valid JSON array format.
"""
    
    def _get_auto_extraction_prompt(self, text: str) -> str:
        return f"""
Analyze this document and extract relevant financial/business data. Determine if it contains:
1. Chart of accounts information
2. Customer/client information  
3. Invoice/billing information
4. Other financial data

Return a JSON object with:
- data_type: "accounts", "customers", "invoices", or "mixed"
- data: Array of extracted objects in appropriate format
- confidence: 0-1 confidence score

Text to analyze:
{text[:4000]}

Return only valid JSON format.
"""

class OAuth2Handler:
    def __init__(self, config: Config):
        self.config = config
    
    def get_authorization_url(self, state: str = None) -> str:
        """Generate QuickBooks OAuth2 authorization URL"""
        if not state:
            state = str(uuid.uuid4())
        
        params = {
            'client_id': self.config.QUICKBOOKS_CLIENT_ID,
            'scope': self.config.QUICKBOOKS_SCOPE,
            'redirect_uri': self.config.QUICKBOOKS_REDIRECT_URI,
            'response_type': 'code',
            'access_type': 'offline',
            'state': state
        }
        
        return f"{self.config.QUICKBOOKS_AUTH_URL}?{urlencode(params)}"
    
    def get_quickbooks_online_url(self, realm_id: str = None) -> str:
        """Generate QuickBooks Online app URL"""
        if realm_id:
            return f"{self.config.QUICKBOOKS_ONLINE_URL}/app/{realm_id}"
        else:
            return self.config.QUICKBOOKS_ONLINE_URL
    
    def exchange_code_for_tokens(self, authorization_code: str, realm_id: str) -> Dict:
        """Exchange authorization code for access and refresh tokens"""
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'Authorization': f'Basic {self._get_basic_auth_header()}'
        }
        
        data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': self.config.QUICKBOOKS_REDIRECT_URI
        }
        
        try:
            response = requests.post(
                self.config.QUICKBOOKS_TOKEN_URL,
                headers=headers,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                token_data = response.json()
                return {
                    'success': True,
                    'access_token': token_data.get('access_token'),
                    'refresh_token': token_data.get('refresh_token'),
                    'expires_in': token_data.get('expires_in', 3600),
                    'token_type': token_data.get('token_type', 'Bearer'),
                    'realm_id': realm_id
                }
            else:
                return {'success': False, 'error': f'Token exchange failed: {response.text}'}
        except Exception as e:
            return {'success': False, 'error': f'Token exchange error: {str(e)}'}
    
    def refresh_access_token(self, refresh_token: str) -> Dict:
        """Refresh access token using refresh token"""
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'Authorization': f'Basic {self._get_basic_auth_header()}'
        }
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token
        }
        
        try:
            response = requests.post(
                self.config.QUICKBOOKS_TOKEN_URL,
                headers=headers,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                token_data = response.json()
                return {
                    'success': True,
                    'access_token': token_data.get('access_token'),
                    'refresh_token': token_data.get('refresh_token'),
                    'expires_in': token_data.get('expires_in', 3600)
                }
            else:
                return {'success': False, 'error': f'Token refresh failed: {response.text}'}
        except Exception as e:
            return {'success': False, 'error': f'Token refresh error: {str(e)}'}
    
    def _get_basic_auth_header(self) -> str:
        """Generate Basic Auth header for OAuth2 requests"""
        credentials = f"{self.config.QUICKBOOKS_CLIENT_ID}:{self.config.QUICKBOOKS_CLIENT_SECRET}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return encoded_credentials

class TokenManager:
    def __init__(self, config: Config, oauth_handler: OAuth2Handler):
        self.config = config
        self.oauth_handler = oauth_handler
        self.access_token = None
        self.refresh_token = None
        self.realm_id = None
        self.token_expires_at = None
        self.lock = threading.Lock()
    
    def set_tokens(self, access_token: str, refresh_token: str, realm_id: str, expires_in: int = 3600):
        """Set the OAuth tokens"""
        with self.lock:
            self.access_token = access_token
            self.refresh_token = refresh_token
            self.realm_id = realm_id
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 300)
    
    def is_token_expired(self) -> bool:
        """Check if token is expired or will expire soon"""
        if not self.token_expires_at:
            return True
        return datetime.now() >= self.token_expires_at
    
    def get_valid_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary"""
        if not self.access_token:
            return None
        
        if self.is_token_expired() and self.refresh_token:
            refresh_result = self.oauth_handler.refresh_access_token(self.refresh_token)
            if refresh_result['success']:
                self.set_tokens(
                    refresh_result['access_token'],
                    refresh_result.get('refresh_token', self.refresh_token),
                    self.realm_id,
                    refresh_result['expires_in']
                )
            else:
                return None
        
        return self.access_token
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated with valid tokens"""
        return self.access_token is not None and self.realm_id is not None
    
    def get_quickbooks_url(self) -> str:
        """Get QuickBooks Online URL for the connected company"""
        if self.realm_id:
            return self.oauth_handler.get_quickbooks_online_url(self.realm_id)
        return self.oauth_handler.get_quickbooks_online_url()

class QuickBooksAccountAPI:
    def __init__(self, config: Config, token_manager: TokenManager):
        self.config = config
        self.token_manager = token_manager
    
    def _get_headers(self) -> Optional[Dict]:
        """Get headers with valid access token"""
        token = self.token_manager.get_valid_token()
        if not token:
            return None
        
        return {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make API request to QuickBooks"""
        headers = self._get_headers()
        if not headers:
            return {"error": "Authentication required. Please connect to QuickBooks."}
        
        url = f"{self.config.QUICKBOOKS_BASE_URL}/v3/company/{self.token_manager.realm_id}/{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=headers, json=data, timeout=30)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
            
            if response.status_code in [200, 201]:
                return response.json()
            else:
                return {"error": f"API Error {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def create_account(self, account_data: Dict) -> Dict:
        """Create a new account in QuickBooks"""
        account_payload = {
            "Account": {
                "Name": account_data.get("name", ""),
                "AccountType": account_data.get("account_type", "Expense"),
                "AccountSubType": account_data.get("account_subtype", "OtherMiscellaneousExpense"),
                "Description": account_data.get("description", ""),
                "Active": account_data.get("active", True)
            }
        }
        
        if "opening_balance" in account_data and account_data["opening_balance"]:
            account_payload["Account"]["OpeningBalance"] = account_data["opening_balance"]
            account_payload["Account"]["OpeningBalanceDate"] = account_data.get(
                "opening_balance_date", 
                datetime.now().strftime("%Y-%m-%d")
            )
        
        return self._make_request('POST', 'account', account_payload)
    
    def get_accounts(self, account_type: str = None, active_only: bool = True) -> Dict:
        """Get list of accounts from QuickBooks"""
        query = "SELECT * FROM Account"
        
        conditions = []
        if account_type:
            conditions.append(f"AccountType = '{account_type}'")
        if active_only:
            conditions.append("Active = true")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " MAXRESULTS 100"
        
        return self._make_request('GET', f'query?query={query}')
    
    def create_customer(self, customer_data: Dict) -> Dict:
        """Create a customer in QuickBooks"""
        customer_payload = {
            "Customer": {
                "Name": customer_data.get("name", ""),
                "CompanyName": customer_data.get("company", ""),
                "BillAddr": {
                    "Line1": customer_data.get("address_line1", ""),
                    "City": customer_data.get("city", ""),
                    "Country": customer_data.get("country", ""),
                    "PostalCode": customer_data.get("postal_code", "")
                },
                "PrimaryEmailAddr": {
                    "Address": customer_data.get("email", "")
                },
                "PrimaryPhone": {
                    "FreeFormNumber": customer_data.get("phone", "")
                }
            }
        }
        
        return self._make_request('POST', 'customer', customer_payload)
    
    def create_invoice(self, invoice_data: Dict) -> Dict:
        """Create an invoice in QuickBooks"""
        customer_ref = invoice_data.get("customer_ref")
        if not customer_ref:
            return {"error": "Customer reference is required for invoice creation"}
        
        invoice_payload = {
            "Invoice": {
                "CustomerRef": {"value": customer_ref},
                "Line": []
            }
        }
        
        for item in invoice_data.get("items", []):
            line = {
                "Amount": item.get("amount", 0),
                "DetailType": "SalesItemLineDetail",
                "SalesItemLineDetail": {
                    "ItemRef": {"value": item.get("item_ref", "1")},
                    "Qty": item.get("quantity", 1),
                    "UnitPrice": item.get("unit_price", 0)
                }
            }
            invoice_payload["Invoice"]["Line"].append(line)
        
        if "due_date" in invoice_data:
            invoice_payload["Invoice"]["DueDate"] = invoice_data["due_date"]
        
        return self._make_request('POST', 'invoice', invoice_payload)
    
    def test_connection(self) -> Dict:
        """Test QuickBooks API connection"""
        if not self.token_manager.is_authenticated():
            return {"connected": False, "error": "Not authenticated. Please connect to QuickBooks."}
        
        result = self._make_request('GET', f'companyinfo/{self.token_manager.realm_id}')
        if "error" in result:
            return {"connected": False, "error": result["error"]}
        
        company_info = result.get("QueryResponse", {}).get("CompanyInfo", [{}])[0]
        return {
            "connected": True, 
            "company": company_info.get("CompanyName", "Unknown"),
            "realm_id": self.token_manager.realm_id
        }

class EnhancedLLMAgent:
    def __init__(self, config: Config):
        self.config = config
        self.system_prompt = """You are Novia, an AI Virtual Accountant assistant specializing in QuickBooks Online integration with advanced document processing capabilities. You help users with:
        
        CORE FUNCTIONS:
        1. Processing documents (PDFs, images, spreadsheets) to extract financial data
        2. Creating and managing Chart of Accounts in QuickBooks
        3. Setting up customers, vendors, and items from extracted data
        4. Creating invoices from scanned receipts and documents
        5. AI-powered data extraction and conversion to QuickBooks format
        6. QuickBooks OAuth authentication and direct access
        
        DOCUMENT PROCESSING CAPABILITIES:
        - PDF text extraction and financial data recognition
        - OCR processing of images, receipts, and scanned documents
        - Excel/CSV data analysis and structure detection
        - AI-powered extraction of accounts, customers, and invoices
        - Bulk import and creation in QuickBooks
        
        ACCOUNTING EXPERTISE:
        - Chart of Accounts setup and management
        - Receipt and invoice processing
        - Customer data extraction from business cards/documents
        - Financial statement preparation concepts
        - Bookkeeping automation
        
        TECHNICAL CAPABILITIES:
        - QuickBooks Online API integration with direct app access
        - Multi-format document processing (PDF, images, spreadsheets)
        - AI-powered data extraction and structure recognition
        - Real-time financial data synchronization
        - Bulk operations and batch processing
        
        CONVERSATION STYLE:
        - Professional but friendly and approachable
        - Explain document processing capabilities clearly
        - Guide users through upload and extraction process
        - Celebrate successful data imports and processing
        - Patient with users learning document processing workflows
        
        When users mention opening QuickBooks or accessing their QuickBooks account, offer to redirect them directly to their QuickBooks Online application.
        
        Always prioritize accuracy in financial matters and data extraction results."""
        
        self.conversations = {}
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        return self.conversations[session_id]
    
    def add_to_conversation(self, session_id: str, role: str, content: str):
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({"role": role, "content": content})
        
        if len(self.conversations[session_id]) > 30:
            self.conversations[session_id] = self.conversations[session_id][-30:]
    
    def get_response(self, user_message: str, session_id: str, context: Dict = None) -> str:
        """Get response from LLM"""
        if not self.config.GROQ_API_KEY or self.config.GROQ_API_KEY in ["your_groq_api_key_here", "", "sk-"]:
            return "Please set a valid Groq API key in your environment variables to enable AI responses. You can get one from https://console.groq.com/"
        
        headers = {
            "Authorization": f"Bearer {self.config.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        enhanced_message = user_message
        if context:
            context_info = []
            if context.get('authenticated'):
                context_info.append("QuickBooks connected")
            if context.get('accounts'):
                context_info.append(f"{len(context['accounts'])} accounts in chart")
            if context.get('processed_documents'):
                context_info.append(f"{len(context['processed_documents'])} documents processed")
            
            if context_info:
                enhanced_message += f"\n\nStatus: {', '.join(context_info)}"
        
        conversation_history = self.get_conversation_history(session_id)
        messages = [
            {"role": "system", "content": self.system_prompt},
            *conversation_history[-10:],
            {"role": "user", "content": enhanced_message}
        ]
        
        payload = {
            "model": self.config.MODEL_NAME,
            "messages": messages,
            "max_tokens": 1200,
            "temperature": 0.3,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                self.config.GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]
                
                self.add_to_conversation(session_id, "user", user_message)
                self.add_to_conversation(session_id, "assistant", assistant_message)
                
                return assistant_message
            else:
                return f"I'm having trouble connecting to my AI service right now. Please try again in a moment. (Error: {response.status_code})"
        except Exception as e:
            return f"I encountered an unexpected issue. Please try rephrasing your question. 🤔"
    
    def clear_conversation(self, session_id: str):
        if session_id in self.conversations:
            self.conversations[session_id] = []

# Enhanced Excel Processor
class ExcelProcessor:
    @staticmethod
    def process_account_data(file_path: str) -> List[Dict]:
        """Process Excel file containing chart of accounts data"""
        try:
            df = pd.read_excel(file_path)
            
            accounts = []
            for _, row in df.iterrows():
                account = {
                    "name": row.get("Account_Name", row.get("Name", "")),
                    "account_type": row.get("Account_Type", "Expense"),
                    "account_subtype": row.get("Account_Subtype", "OtherMiscellaneousExpense"),
                    "description": row.get("Description", ""),
                    "opening_balance": row.get("Opening_Balance", 0) if pd.notna(row.get("Opening_Balance")) else None,
                    "opening_balance_date": row.get("Opening_Balance_Date", datetime.now().strftime("%Y-%m-%d")),
                    "active": row.get("Active", True)
                }
                accounts.append(account)
            
            return accounts
        except Exception as e:
            raise Exception(f"Error processing accounts Excel file: {str(e)}")
    
    @staticmethod
    def process_invoice_data(file_path: str) -> List[Dict]:
        """Process Excel file containing invoice data"""
        try:
            df = pd.read_excel(file_path)
            
            invoices = []
            for _, row in df.iterrows():
                invoice = {
                    "customer_name": row.get("Customer", ""),
                    "customer_ref": row.get("Customer_ID", "1"),
                    "items": [{
                        "description": row.get("Description", ""),
                        "quantity": row.get("Quantity", 1),
                        "unit_price": row.get("Unit_Price", 0),
                        "amount": row.get("Total", 0),
                        "item_ref": row.get("Item_ID", "1")
                    }],
                    "total_amount": row.get("Total", 0),
                    "due_date": row.get("Due_Date", (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
                }
                invoices.append(invoice)
            
            return invoices
        except Exception as e:
            raise Exception(f"Error processing invoice Excel file: {str(e)}")
    
    @staticmethod
    def process_customer_data(file_path: str) -> List[Dict]:
        """Process Excel file containing customer data"""
        try:
            df = pd.read_excel(file_path)
            
            customers = []
            for _, row in df.iterrows():
                customer = {
                    "name": row.get("Name", ""),
                    "company": row.get("Company", ""),
                    "email": row.get("Email", ""),
                    "address_line1": row.get("Address", ""),
                    "city": row.get("City", ""),
                    "postal_code": row.get("Postal_Code", ""),
                    "country": row.get("Country", "USA"),
                    "phone": row.get("Phone", "")
                }
                customers.append(customer)
            
            return customers
        except Exception as e:
            raise Exception(f"Error processing customer Excel file: {str(e)}")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', str(uuid.uuid4()))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max for document processing

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
config = Config()
oauth_handler = OAuth2Handler(config)
token_manager = TokenManager(config, oauth_handler)
qb_api = QuickBooksAccountAPI(config, token_manager)
llm_agent = EnhancedLLMAgent(config)
excel_processor = ExcelProcessor()
document_processor = DocumentProcessor(config)

# Session data storage
session_data = {}

def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_session_data(session_id: str) -> Dict:
    """Get session data"""
    if session_id not in session_data:
        session_data[session_id] = {
            'accounts': [],
            'customers': [],
            'invoices': [],
            'processed_documents': [],
            'chat_history': []
        }
    return session_data[session_id]

# Routes
@app.route('/')
def index():
    """Main dashboard"""
    session_id = get_session_id()
    auth_status = token_manager.is_authenticated()
    return render_template('index.html', 
                         session_id=session_id, 
                         authenticated=auth_status,
                         auth_url=oauth_handler.get_authorization_url() if not auth_status else None,
                         quickbooks_url=token_manager.get_quickbooks_url() if auth_status else None)

@app.route('/connect')
def connect_quickbooks():
    """Initiate QuickBooks OAuth flow"""
    auth_url = oauth_handler.get_authorization_url()
    return redirect(auth_url)

@app.route('/callback')
def oauth_callback():
    """Handle OAuth callback from QuickBooks"""
    code = request.args.get('code')
    realm_id = request.args.get('realmId')
    error = request.args.get('error')
    
    if error:
        return f"<h1>Authorization Error</h1><p>{error}</p><a href='/'>Go back</a>"
    
    if not code or not realm_id:
        return f"<h1>Authorization Error</h1><p>Missing authorization code or company ID</p><a href='/'>Go back</a>"
    
    # Exchange code for tokens
    token_result = oauth_handler.exchange_code_for_tokens(code, realm_id)
    
    if token_result['success']:
        token_manager.set_tokens(
            token_result['access_token'],
            token_result['refresh_token'],
            token_result['realm_id'],
            token_result['expires_in']
        )
        return redirect(url_for('index'))
    else:
        return f"<h1>Token Exchange Error</h1><p>{token_result['error']}</p><a href='/'>Go back</a>"

@app.route('/disconnect')
def disconnect_quickbooks():
    """Disconnect from QuickBooks"""
    token_manager.access_token = None
    token_manager.refresh_token = None
    token_manager.realm_id = None
    token_manager.token_expires_at = None
    return redirect(url_for('index'))

@app.route('/open-quickbooks')
def open_quickbooks():
    """Redirect to QuickBooks Online"""
    if token_manager.is_authenticated():
        qb_url = token_manager.get_quickbooks_url()
        return redirect(qb_url)
    else:
        return redirect(url_for('connect_quickbooks'))

# API Routes
@app.route('/api/status')
def get_status():
    """Get connection status and stats"""
    try:
        if not token_manager.is_authenticated():
            return jsonify({
                "connected": False,
                "error": "Not connected to QuickBooks",
                "auth_url": oauth_handler.get_authorization_url(),
                "quickbooks_url": None,
                "stats": {"accounts": 0, "customers": 0, "invoices": 0, "documents": 0}
            })
        
        connection_result = qb_api.test_connection()
        
        if connection_result.get("connected"):
            accounts_result = qb_api.get_accounts()
            
            stats = {"accounts": 0, "customers": 0, "invoices": 0, "documents": 0}
            
            if "error" not in accounts_result:
                accounts = accounts_result.get("QueryResponse", {}).get("Account", [])
                stats["accounts"] = len(accounts)
            
            # Get document processing stats
            session_id = get_session_id()
            session_data_obj = get_session_data(session_id)
            stats["documents"] = len(session_data_obj.get('processed_documents', []))
            
            return jsonify({
                "connected": True,
                "company": connection_result.get("company"),
                "realm_id": connection_result.get("realm_id"),
                "quickbooks_url": token_manager.get_quickbooks_url(),
                "stats": stats
            })
        else:
            return jsonify({
                "connected": False,
                "error": connection_result.get("error"),
                "auth_url": oauth_handler.get_authorization_url(),
                "quickbooks_url": None,
                "stats": {"accounts": 0, "customers": 0, "invoices": 0, "documents": 0}
            })
    except Exception as e:
        return jsonify({
            "connected": False,
            "error": str(e),
            "auth_url": oauth_handler.get_authorization_url(),
            "quickbooks_url": None,
            "stats": {"accounts": 0, "customers": 0, "invoices": 0, "documents": 0}
        })

@app.route('/api/accounts', methods=['GET'])
def get_accounts():
    """Get QuickBooks accounts"""
    try:
        account_type = request.args.get('type')
        result = qb_api.get_accounts(account_type)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        accounts = result.get("QueryResponse", {}).get("Account", [])
        return jsonify({"accounts": accounts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/accounts', methods=['POST'])
def create_account():
    """Create a new account"""
    try:
        account_data = request.json
        result = qb_api.create_account(account_data)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        return jsonify({"success": True, "account": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/customers', methods=['GET'])
def get_customers():
    """Get QuickBooks customers"""
    try:
        result = qb_api._make_request('GET', 'query?query=SELECT * FROM Customer MAXRESULTS 100')
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        customers = result.get("QueryResponse", {}).get("Customer", [])
        return jsonify({"customers": customers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/customers', methods=['POST'])
def create_customer():
    """Create a new customer"""
    try:
        customer_data = request.json
        result = qb_api.create_customer(customer_data)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        return jsonify({"success": True, "customer": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/invoices', methods=['GET'])
def get_invoices():
    """Get QuickBooks invoices"""
    try:
        result = qb_api._make_request('GET', 'query?query=SELECT * FROM Invoice MAXRESULTS 100')
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        invoices = result.get("QueryResponse", {}).get("Invoice", [])
        return jsonify({"invoices": invoices})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/invoices', methods=['POST'])
def create_invoice():
    """Create a new invoice"""
    try:
        invoice_data = request.json
        result = qb_api.create_invoice(invoice_data)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        return jsonify({"success": True, "invoice": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    """Handle advanced document uploads with AI processing"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    processing_type = request.form.get('type', 'auto')  # auto, accounts, customers, invoices
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not document_processor.is_allowed_file(file.filename):
        return jsonify({"error": f"File type not supported. Allowed types: {', '.join(config.ALLOWED_EXTENSIONS)}"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{get_session_id()}_{filename}")
    file.save(filepath)
    
    try:
        session_id = get_session_id()
        session_data_obj = get_session_data(session_id)
        
        file_extension = Path(filepath).suffix.lower()
        extracted_text = ""
        extracted_data = []
        processing_method = ""
        
        # Process based on file type
        if file_extension == '.pdf':
            processing_method = "PDF text extraction"
            extracted_text = document_processor.extract_text_from_pdf(filepath)
            
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            processing_method = "OCR image processing"
            extracted_text = document_processor.extract_text_from_image(filepath)
            
        elif file_extension in ['.xlsx', '.xls', '.csv']:
            processing_method = "Spreadsheet analysis"
            if processing_type == 'accounts':
                extracted_data = excel_processor.process_account_data(filepath)
            elif processing_type == 'customers':
                extracted_data = excel_processor.process_customer_data(filepath)
            elif processing_type == 'invoices':
                extracted_data = excel_processor.process_invoice_data(filepath)
            else:
                # Use AI to analyze spreadsheet data
                spreadsheet_data = document_processor.extract_data_from_spreadsheet(filepath)
                extracted_text = f"Spreadsheet with {spreadsheet_data['shape'][0]} rows and {spreadsheet_data['shape'][1]} columns.\n"
                extracted_text += f"Columns: {', '.join(spreadsheet_data['columns'])}\n"
                extracted_text += f"Sample data: {str(spreadsheet_data['rows'][:5])}"
        
        # Use AI extraction for text-based content
        if extracted_text and not extracted_data:
            processing_method += " + AI extraction"
            extracted_data = document_processor.ai_extract_structured_data(extracted_text, processing_type)
        
        # Store processed data
        if extracted_data:
            # Determine data type if auto
            if processing_type == 'auto':
                # Simple heuristic to determine data type
                sample_item = extracted_data[0] if extracted_data else {}
                if 'account_type' in sample_item or 'account_subtype' in sample_item:
                    processing_type = 'accounts'
                elif 'customer_name' in sample_item or 'invoice_number' in sample_item:
                    processing_type = 'invoices'
                elif 'email' in sample_item or 'phone' in sample_item:
                    processing_type = 'customers'
            
            # Store in session
            if processing_type in ['accounts', 'customers', 'invoices']:
                session_data_obj[processing_type].extend(extracted_data)
            
            # Track processed document
            doc_info = {
                'filename': filename,
                'type': processing_type,
                'method': processing_method,
                'extracted_count': len(extracted_data),
                'timestamp': datetime.now().isoformat(),
                'preview': extracted_text[:200] + "..." if extracted_text else ""
            }
            session_data_obj['processed_documents'].append(doc_info)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            "success": True,
            "message": f"Successfully processed {filename} using {processing_method}",
            "extracted_count": len(extracted_data),
            "data_type": processing_type,
            "preview": extracted_text[:200] + "..." if extracted_text else "",
            "data": extracted_data[:5]  # Return first 5 items for preview
        })
    
    except Exception as e:
        # Clean up uploaded file on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500

@app.route('/api/bulk-create', methods=['POST'])
def bulk_create():
    """Bulk create accounts, customers, or invoices from processed documents"""
    try:
        data = request.json
        entity_type = data.get('type')  # 'accounts', 'customers', 'invoices'
        session_id = get_session_id()
        session_data_obj = get_session_data(session_id)
        
        if entity_type not in session_data_obj or not session_data_obj[entity_type]:
            return jsonify({"error": f"No {entity_type} data found. Please process a document first."}), 400
        
        created_items = []
        errors = []
        
        for item in session_data_obj[entity_type]:
            try:
                if entity_type == 'accounts':
                    result = qb_api.create_account(item)
                elif entity_type == 'customers':
                    result = qb_api.create_customer(item)
                elif entity_type == 'invoices':
                    result = qb_api.create_invoice(item)
                else:
                    continue
                
                if "error" in result:
                    errors.append(f"{item.get('name', 'Unknown')}: {result['error']}")
                else:
                    created_items.append(result)
            except Exception as e:
                errors.append(f"{item.get('name', 'Unknown')}: {str(e)}")
        
        return jsonify({
            "success": True,
            "created": len(created_items),
            "errors": errors,
            "total": len(session_data_obj[entity_type])
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/processed-data/<data_type>')
def get_processed_data(data_type):
    """Get processed data for preview/editing"""
    try:
        session_id = get_session_id()
        session_data_obj = get_session_data(session_id)
        
        if data_type not in session_data_obj:
            return jsonify({"error": "Invalid data type"}), 400
        
        return jsonify({
            "success": True,
            "data": session_data_obj[data_type],
            "count": len(session_data_obj[data_type])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear-processed-data', methods=['POST'])
def clear_processed_data():
    """Clear processed document data"""
    try:
        data = request.json
        data_type = data.get('type', 'all')
        session_id = get_session_id()
        session_data_obj = get_session_data(session_id)
        
        if data_type == 'all':
            session_data_obj['accounts'] = []
            session_data_obj['customers'] = []
            session_data_obj['invoices'] = []
            session_data_obj['processed_documents'] = []
        elif data_type in session_data_obj:
            session_data_obj[data_type] = []
        
        return jsonify({"success": True, "message": f"Cleared {data_type} data"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    try:
        session_id = get_session_id()
        print(f"Client connected: {session_id}")
        emit('connected', {
            'session_id': session_id,
            'authenticated': token_manager.is_authenticated(),
            'quickbooks_url': token_manager.get_quickbooks_url() if token_manager.is_authenticated() else None
        })
    except Exception as e:
        print(f"Connection error: {str(e)}")
        emit('error', {'message': 'Connection failed'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('send_message')
def handle_message(data):
    """Handle chat messages with enhanced context"""
    try:
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', get_session_id())
        
        if not user_message:
            emit('error', {'message': 'Empty message'})
            return
        
        print(f"Processing message: {user_message[:50]}...")
        
        # Get enhanced context including document processing data
        context = get_session_data(session_id)
        context['authenticated'] = token_manager.is_authenticated()
        
        if token_manager.is_authenticated():
            try:
                accounts_result = qb_api.get_accounts()
                if "error" not in accounts_result:
                    context['accounts'] = accounts_result.get("QueryResponse", {}).get("Account", [])
            except Exception as e:
                print(f"Error getting context data: {str(e)}")
        
        # Check if user wants to open QuickBooks
        if any(phrase in user_message.lower() for phrase in ['open quickbooks', 'go to quickbooks', 'launch quickbooks', 'access quickbooks']):
            if token_manager.is_authenticated():
                qb_url = token_manager.get_quickbooks_url()
                response = f"I'll open QuickBooks Online for you! You can access it at: {qb_url}"
                emit('quickbooks_redirect', {'url': qb_url})
            else:
                response = "You need to connect to QuickBooks first. Let me help you set that up!"
                emit('auth_required', {'auth_url': oauth_handler.get_authorization_url()})
            
            emit('message_response', {
                'message': response,
                'sender': 'Novia',
                'timestamp': datetime.now().strftime('%H:%M')
            })
        else:
            # Get AI response
            try:
                response = llm_agent.get_response(user_message, session_id, context)
                emit('message_response', {
                    'message': response,
                    'sender': 'Novia',
                    'timestamp': datetime.now().strftime('%H:%M')
                })
            except Exception as e:
                print(f"Error getting AI response: {str(e)}")
                emit('message_response', {
                    'message': f"I apologize, but I encountered an error while processing your request: {str(e)}",
                    'sender': 'Novia',
                    'timestamp': datetime.now().strftime('%H:%M')
                })
        
        # Acknowledge user message
        emit('message_sent', {
            'message': user_message,
            'sender': 'You',
            'timestamp': datetime.now().strftime('%H:%M')
        })
        
    except Exception as e:
        print(f"Message handling error: {str(e)}")
        emit('error', {'message': f'Message processing failed: {str(e)}'})

@socketio.on('clear_chat')
def handle_clear_chat(data):
    """Clear chat history"""
    try:
        session_id = data.get('session_id', get_session_id())
        llm_agent.clear_conversation(session_id)
        emit('chat_cleared')
        print(f"Chat cleared for session: {session_id}")
    except Exception as e:
        print(f"Clear chat error: {str(e)}")
        emit('error', {'message': 'Failed to clear chat'})

@socketio.on('get_auth_status')
def handle_auth_status():
    """Get current authentication status"""
    try:
        emit('auth_status', {
            'authenticated': token_manager.is_authenticated(),
            'auth_url': oauth_handler.get_authorization_url() if not token_manager.is_authenticated() else None,
            'quickbooks_url': token_manager.get_quickbooks_url() if token_manager.is_authenticated() else None
        })
    except Exception as e:
        print(f"Auth status error: {str(e)}")
        emit('error', {'message': 'Failed to get auth status'})

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large. Maximum size is 50MB."}), 413

if __name__ == '__main__':
    # Check for required environment variables and dependencies
    required_vars = [
        "QUICKBOOKS_CLIENT_ID", 
        "QUICKBOOKS_CLIENT_SECRET", 
        "GROQ_API_KEY"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    # Check for optional dependencies
    optional_deps = []
    try:
        import fitz
    except ImportError:
        optional_deps.append("PyMuPDF (for PDF processing)")
    
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except:
        optional_deps.append("Tesseract OCR (for image text extraction)")
    
    try:
        import cv2
    except ImportError:
        optional_deps.append("OpenCV (for advanced image processing)")
    
    if missing_vars:
        print("⚠️  Warning: Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nTo get started:")
        print("1. Get QuickBooks app credentials from https://developer.intuit.com")
        print("2. Get Groq API key from https://console.groq.com")
        print("3. Set environment variables:")
        print("   export QUICKBOOKS_CLIENT_ID='your_client_id'")
        print("   export QUICKBOOKS_CLIENT_SECRET='your_client_secret'")
        print("   export QUICKBOOKS_REDIRECT_URI='http://localhost:5000/callback'")
        print("   export GROQ_API_KEY='your_groq_api_key'")
    
    if optional_deps:
        print("⚠️  Optional dependencies missing (document processing will be limited):")
        for dep in optional_deps:
            print(f"   - {dep}")
        print("\nInstall with:")
        print("   pip install PyMuPDF pytesseract opencv-python")
        print("   # Also install Tesseract OCR system package")
    
    if not missing_vars:
        print("✅ All required environment variables are set!")
    
    print(f"🚀 Starting Enhanced Novia - AI Document Processing & QuickBooks Integration")
    print(f"📱 Dashboard: http://localhost:5000")
    print(f"🔗 OAuth Callback: {config.QUICKBOOKS_REDIRECT_URI}")
    print(f"📄 Supported formats: PDFs, Images (PNG/JPG/TIFF), Spreadsheets (Excel/CSV)")
    print(f"🤖 AI-powered data extraction and QuickBooks integration")
    
    # Run the application
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)