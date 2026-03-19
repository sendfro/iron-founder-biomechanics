import json
import time

# Here we import the Eyes, the Brain, and the Voice that you just built!
from extractor import extract_invoice_data
from auditor import run_audit
from disputer import generate_dispute_email

def run_full_pipeline(image_path):
    print("\n========================================================")
    print(" 🚀 INITIATING AUTOMATED FREIGHT PIPELINE 🚀")
    print("========================================================\n")

    # --- PHASE 1: THE EYES ---
    print(">>> STEP 1: EXTRACTING DATA FROM INVOICE...")
    extracted_data = extract_invoice_data(image_path)
    print("    [+] Extraction Complete.\n")
    time.sleep(1) # Just a slight pause so we can read the terminal output

    # We build a temporary internal database for whatever invoice number it just found
    # We will tell our system we only expected to pay $1100.00
    simulated_db = {
        extracted_data.get("invoice_number", "UNKNOWN"): {
            "expected_total": 1100.0,
            "approved_fees": []
        }
    }

    # --- PHASE 2: THE BRAIN ---
    print(">>> STEP 2: RUNNING RECONCILIATION AUDIT...")
    audit_report = run_audit(extracted_data, simulated_db)
    print(f"    [+] Audit Complete. Status: {audit_report['status']}\n")
    time.sleep(1)

    # --- PHASE 3: THE VOICE ---
    # The machine makes a decision: do we fight, or do we pay?
    if audit_report["status"] == "DISPUTE REQUIRED":
        print(">>> STEP 3: DISPUTE TRIGGERED. DRAFTING CARRIER EMAIL...")
        
        invoice_details = {
            "carrier_name": extracted_data["carrier_name"],
            "invoice_number": extracted_data["invoice_number"]
        }
        
        email_draft = generate_dispute_email(audit_report, invoice_details)
        
        print("\n================ DRAFTED EMAIL ================\n")
        print(email_draft)
        print("\n===============================================\n")
    else:
         print(">>> STEP 3: INVOICE IS CLEAN. NO DISPUTE NEEDED. APPROVED FOR PAYMENT.\n")

    print(" 🚀 PIPELINE EXECUTION FINISHED 🚀\n")

# --- Execution ---
if __name__ == "__main__":
    # This is the fuel for our master engine
    target_invoice = "carrier_invoice_sample.jpg"
    
    try:
        run_full_pipeline(target_invoice)
    except Exception as e:
        print(f"\n[-] Pipeline Failed: {e}")