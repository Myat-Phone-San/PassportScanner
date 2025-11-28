import os
import streamlit as st
import re
from datetime import datetime
import pandas as pd
from io import BytesIO
import time 
import json 
from google import genai
from google.genai import types
from PIL import Image

# --- Configuration ---
st.set_page_config(
    page_title="🛂 Myanmar Passport Extractor (AI OCR) with Validation",
    layout="wide"
)

# Initialize the Gemini Client
try:
    # Use st.secrets to securely load the API key
    api_key = st.secrets["GEMINI_API_KEY"] 
    client = genai.Client(api_key=api_key) 
except KeyError:
    st.error("Error: GEMINI_API_KEY not found in Streamlit Secrets. Please configure your secrets file/settings.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing AI client. Details: {e}")
    st.stop()


# --- 1. MRZ Checksum Validation Logic (No change) ---

# Lookup table for character values (A=10, B=11, ..., Z=35, < or space = 0)
MRZ_CHAR_VALUES = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '<': 0, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 
    'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 
    'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35
}
# Weights used in the Modulo 10 algorithm
WEIGHTS = [7, 3, 1]

def calculate_mrz_checksum(data_string: str) -> str:
    """
    Calculates the checksum digit for a given MRZ data field 
    (Passport No, Date of Birth, Date of Expiry) using the Modulo 10 algorithm.
    """
    total_sum = 0
    
    # 1. Iterate through the string, applying weights (7, 3, 1, 7, 3, 1...)
    for i, char in enumerate(data_string.upper()):
        value = MRZ_CHAR_VALUES.get(char, 0)
        weight = WEIGHTS[i % 3]
        total_sum += value * weight
        
    # 3. The checksum is the remainder of the total sum when divided by 10 (Modulo 10)
    checksum = total_sum % 10
    
    return str(checksum)

# --- 2. Data Extraction Prompt and Schema (UPDATED FOR SINGLE MRZ VALUE) ---

# Define the expected output structure for a Myanmar Passport
extraction_schema = {
    "type": "object",
    "properties": {
        # Primary English/Latin Script Fields 
        "type": {"type": "string", "description": "The passport type, e.g., 'PV' (Private)."},
        "country_code": {"type": "string", "description": "The country code, e.g., 'MMR'."},
        "passport_no": {"type": "string", "description": "The passport number (e.g., MH000000)."},
        "name": {"type": "string", "description": "The full name of the passport holder in Latin script (e.g., MIN ZAW)."},
        "nationality": {"type": "string", "description": "The nationality (e.g., MYANMAR)."},
        "date_of_birth": {"type": "string", "description": "The date of birth in DD-MM-YYYY format."},
        "sex": {"type": "string", "description": "The sex/gender, e.g., 'M' or 'F'."},
        "place_of_birth": {"type": "string", "description": "The place of birth (e.g., SAGAING)."},
        "date_of_issue": {"type": "string", "description": "The date of issue in DD-MM-YYYY format."},
        "date_of_expiry": {"type": "string", "description": "The date of expiry in DD-MM-YYYY format."},
        "authority": {"type": "string", "description": "The issuing authority (e.g., MOHA, YANGON)."},
        
        # Machine Readable Zone (MRZ) - NOW COMBINED
        "mrz_full_string": {"type": "string", "description": "The two lines of the Machine Readable Zone (MRZ) combined into one string, separated by a space."},
        "passport_no_checksum": {"type": "string", "description": "The single checksum digit corresponding to the Passport No in the MRZ."},

        "extraction_confidence": {"type": "number", "description": "The model's self-assessed confidence score for the entire extraction, from 0.0 (low) to 1.0 (high)."}
    },
    "required": [
        "type", "country_code", "passport_no", "name", "nationality", 
        "date_of_birth", "sex", "place_of_birth", "date_of_issue", 
        "date_of_expiry", "authority", "mrz_full_string", 
        "passport_no_checksum", "extraction_confidence"
    ]
}

# The main prompt for the model (UPDATED FOR SINGLE MRZ VALUE)
EXTRACTION_PROMPT = """
Analyze the provided image, which is a Myanmar Passport (Biographical Data Page).
Extract ALL data fields shown on the page and the Machine Readable Zone (MRZ).

Return the result strictly as a JSON object matching the provided schema.

1.  **Main Fields**: Extract Type, Country code, Passport No, Name, Nationality, Date of Birth, Sex, Place of birth, Date of issue, Date of expiry, and Authority.
2.  **Date Format**: Ensure all dates are converted to the **DD-MM-YYYY** format (e.g., 17 JAN 2023 -> 17-01-2023).
3.  **MRZ**: Extract the two full lines of the Machine Readable Zone (MRZ) at the bottom and combine them into a single string. Separate the two lines with a single space.
4.  **Checksum**: Specifically extract the single digit checksum for the Passport No.
5.  **Confidence**: Provide your best self-assessed confidence for the entire extraction on a scale of 0.0 to 1.0 for 'extraction_confidence'.

If a field is not found, return an empty string "" for that value.
Do not include any extra text or formatting outside of the JSON object.
"""

# --- 3. File Handling Function (No change) ---

def handle_file_to_pil(uploaded_file):
    """Converts uploaded file or bytes to a PIL Image object."""
    if uploaded_file is None:
        return None
        
    file_bytes = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file
    try:
        image_pil = Image.open(BytesIO(file_bytes))
        return image_pil
    except Exception as e:
        st.error(f"Error converting file to image: {e}")
        return None
        
# --- 4. AI Extraction Logic (No change) ---

def run_structured_extraction(image_pil):
    """Uses the AI API to analyze the image and extract structured data."""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[EXTRACTION_PROMPT, image_pil],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=extraction_schema,
                temperature=0.0,
            )
        )
        structured_data = json.loads(response.text)
        return structured_data
        
    except genai.errors.APIError as e:
        st.error(f"AI API Error: Could not process the image. Details: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during AI processing: {e}")
        return None

# --- 5. Helper Functions (Updated for single MRZ field) ---

def create_downloadable_files(extracted_dict, checksum_verified):
    """Formats the extracted data into CSV, TXT, and DOC formats."""
    
    # Determine the status text based on verification result
    verification_status = "VERIFIED (Checksum Matched)" if checksum_verified else "WARNING: CHECKSUM MISMATCH (Potential Forgery/Error)"

    # 1. Prepare display dictionary
    results_dict = {
        "Verification Status": verification_status,
        "Passport Type": extracted_dict.get('type', ''),
        "Country Code": extracted_dict.get('country_code', ''),
        "Passport No": extracted_dict.get('passport_no', ''),
        "Name": extracted_dict.get('name', ''),
        "Nationality": extracted_dict.get('nationality', ''),
        "Date of Birth (DD-MM-YYYY)": extracted_dict.get('date_of_birth', ''),
        "Sex": extracted_dict.get('sex', ''),
        "Place of Birth": extracted_dict.get('place_of_birth', ''),
        "Date of Issue (DD-MM-YYYY)": extracted_dict.get('date_of_issue', ''),
        "Date of Expiry (DD-MM-YYYY)": extracted_dict.get('date_of_expiry', ''),
        "Authority": extracted_dict.get('authority', ''),
        "MRZ Full String": extracted_dict.get('mrz_full_string', ''), # Updated field name
        "Passport No Checksum (Extracted)": extracted_dict.get('passport_no_checksum', ''),
        "Extraction Confidence (0.0 - 1.0)": f"{extracted_dict.get('extraction_confidence', 0.0):.2f}"
    }
    
    # 2. Prepare TXT content
    txt_content = "\n".join([f"{key}: {value}" for key, value in results_dict.items()])
    
    # 3. Prepare DataFrame for CSV
    df = pd.DataFrame(results_dict.items(), columns=['Field', 'Value'])
    
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8') 
    csv_content = csv_buffer.getvalue()
    
    # 4. Prepare DOC content (tab-separated for easy copy-paste)
    doc_content = "\n".join([f"{key}\t{value}" for key, value in results_dict.items()])
    
    return txt_content, csv_content, doc_content, results_dict, verification_status


# --- 6. UI and Execution Flow (Updated for single MRZ field) ---

def process_image_and_display(original_image_pil, unique_key_suffix):
    """Performs AI extraction, runs checksum validation, and displays results."""
    st.subheader("Processing Passport Image...")
    
    with st.spinner("Running AI Structured Extraction (Passport OCR)..."):
        time.sleep(1) 
        
        # 1. Run Structured Extraction
        raw_extracted_data = run_structured_extraction(original_image_pil)
        
        if raw_extracted_data is None:
             st.stop() 
        
        # --- CHECKSUM VALIDATION STEP (Uses the passport_no extracted by AI) ---
        passport_no_data = raw_extracted_data.get('passport_no', '').replace('<', '')
        extracted_checksum = raw_extracted_data.get('passport_no_checksum', '')

        # Calculate the expected checksum
        calculated_checksum = calculate_mrz_checksum(passport_no_data)
        
        # Determine if the extracted checksum matches the calculated one
        checksum_verified = (calculated_checksum == extracted_checksum) and (extracted_checksum != "")
        
        # 2. Prepare data for display/download
        txt_file, csv_file, doc_file, extracted_data, verification_status = create_downloadable_files(raw_extracted_data, checksum_verified)
        
    
    # --- DISPLAY ALERT MESSAGE ---
    if checksum_verified:
        st.success(f"✅ Extraction Complete and Data VERIFIED! Confidence: **{extracted_data['Extraction Confidence (0.0 - 1.0)']}**")
    else:
        st.warning(f"⚠️ **VALIDATION ERROR!** Checksum Mismatch Detected (Possible Forgery or OCR Error).")
        st.error(f"Extracted Checksum: **{extracted_checksum}** | Calculated Checksum: **{calculated_checksum}**")
    
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Uploaded Passport Page")
        st.image(original_image_pil, use_column_width=True) 
        
    with col2:
        st.header("Extraction Results")
        
        # --- Results Form (Display Extracted Fields) ---
        form_key = f"results_form_{unique_key_suffix}"
        with st.form(form_key): 
            # Display the verification status first
            st.text_input("Verification Status", value=verification_status, disabled=True)
            st.markdown("---")
            
            # Primary Fields
            st.text_input("Name", value=extracted_data["Name"])
            st.text_input("Passport No", value=extracted_data["Passport No"])
            st.text_input("Nationality", value=extracted_data["Nationality"])
            st.text_input("Date of Birth (DD-MM-YYYY)", value=extracted_data["Date of Birth (DD-MM-YYYY)"])
            st.text_input("Date of Expiry (DD-MM-YYYY)", value=extracted_data["Date of Expiry (DD-MM-YYYY)"])
            st.text_input("Sex", value=extracted_data["Sex"])
            st.text_input("Place of Birth", value=extracted_data["Place of Birth"])
            st.text_input("Authority", value=extracted_data["Authority"])
            
            st.markdown("---")
            st.subheader("Machine Readable Zone (MRZ) & Details")
            st.text_input("MRZ Full String", value=extracted_data["MRZ Full String"]) # Updated field name
            st.text_input("Passport No Checksum (Extracted)", value=extracted_data["Passport No Checksum (Extracted)"])
            st.text_input("Passport No Checksum (Calculated)", value=calculated_checksum) # Show calculated value
            st.text_input("Confidence Score", value=extracted_data["Extraction Confidence (0.0 - 1.0)"])

            st.form_submit_button("Acknowledge & Validate") 
            
        st.subheader("Download Data")
        
        # --- Download Buttons ---
        st.download_button(
            label="⬇️ Download CSV", 
            data=csv_file, 
            file_name=f"passport_data_verified_{unique_key_suffix}.csv", 
            mime="text/csv", 
            key=f"download_csv_{unique_key_suffix}"
        )
        st.download_button(
            label="⬇️ Download Plain Text", 
            data=txt_file, 
            file_name=f"passport_data_verified_{unique_key_suffix}.txt", 
            mime="text/plain", 
            key=f"download_txt_{unique_key_suffix}" 
        )
        st.download_button(
            label="⬇️ Download Word (.doc)", 
            data=doc_file, 
            file_name=f"passport_data_verified_{unique_key_suffix}.doc", 
            mime="application/msword", 
            key=f"download_doc_{unique_key_suffix}" 
        )

# --- Main App Body (No change) ---

st.title("🛂 Myanmar Passport Extractor (AI OCR) with Validation")
st.caption("Structured data extraction and **Checksum Verification** using Gemini.")

# --- Tab Setup ---
tab1, tab2 = st.tabs(["📷 Live Capture (Scanner)", "⬆️ Upload File"])

current_time_suffix = str(time.time()).replace('.', '') 

# --- Live Capture Tab ---
with tab1:
    st.header("Live Document Capture")
    st.write("Use your device's camera to scan the Passport Biographical Data Page.")
    captured_file = st.camera_input("Place the passport page clearly in the frame and click 'Take Photo'", key="camera_input")
    
    if captured_file is not None:
        image_pil = handle_file_to_pil(captured_file)
        
        if image_pil is not None:
            process_image_and_display(
                image_pil, 
                f"live_{current_time_suffix}"
            )
        else:
            st.error("Could not read the captured image data. Please ensure the camera capture was successful.")

# --- Upload File Tab ---
with tab2:
    st.header("Upload Image File")
    st.write("Upload a clear photo or scan of the Passport Biographical Data Page.")
    uploaded_file = st.file_uploader("Upload Passport Image", type=['jpg', 'png', 'jpeg'], key="file_uploader")
    
    if uploaded_file is not None:
        image_pil = handle_file_to_pil(uploaded_file)
        
        if image_pil is not None:
            process_image_and_display(
                image_pil, 
                f"upload_{current_time_suffix}"
            )
        else:
            st.error("Could not read the uploaded image data. Please ensure the file is a valid image.")
