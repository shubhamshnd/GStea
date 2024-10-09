import streamlit as st
import json
import pandas as pd
import io
import logging
from fuzzywuzzy import fuzz
from collections import defaultdict
from datetime import datetime
import difflib
import numpy as np
import pyperclip
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_b2b_data(b2b_data):
    processed_data = []
    for entry in b2b_data:
        for inv in entry.get('inv', []):
            invoice_data = {
                'ctin': entry.get('ctin'),
                'trdnm': entry.get('trdnm'),
                'dt': inv.get('dt'),
                'val': inv.get('val'),
                'rev': inv.get('rev'),
                'itcavl': inv.get('itcavl'),
                'diffprcnt': inv.get('diffprcnt'),
                'pos': inv.get('pos'),
                'typ': inv.get('typ'),
                'inum': inv.get('inum'),
                'rsn': inv.get('rsn')
            }
            if 'items' in inv and inv['items']:
                item = inv['items'][0]  # Assuming only one item per invoice
                invoice_data.update({
                    'item_sgst': item.get('sgst'),
                    'item_rt': item.get('rt'),
                    'item_num': item.get('num'),
                    'item_txval': item.get('txval'),
                    'item_cgst': item.get('cgst'),
                    'item_cess': item.get('cess'),
                    'item_igst': item.get('igst')
                })
            processed_data.append(invoice_data)
    return processed_data

def clean_and_prepare_data(df):
    logger.info(f"Columns in dataframe: {df.columns}")
    
    # Define all possible numeric columns
    numeric_columns = ['val', 'item_txval', 'item_cgst', 'item_sgst', 'item_igst', 'item_rt',
                       'InvoiceValue', 'TaxableValue', 'CentralTaxAmount', 'StateUTTaxAmount', 'IntegratedTaxAmount',
                       'IntegratedTaxRate', 'CentralTaxRate', 'StateUTTaxRate']
    
    for col in numeric_columns:
        if col in df.columns:
            # First, replace any non-numeric strings with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Then, convert to float
            df[col] = df[col].astype(float)
            
            logger.info(f"Converted {col} to float. Sample values: {df[col].head()}")
    
    # Fill NaN values with 0 for numeric columns
    df = df.fillna(0)
    
    return df

def preprocess_invoice_number(invoice_number):
    return str(invoice_number).upper().replace(' ', '').replace('-', '').replace('/', '')

def preprocess_supplier_name(name):
    suffixes_to_remove = ['PVT', 'LTD', 'LIMITED', 'PRIVATE']
    name = ' '.join(word for word in str(name).upper().split() if word not in suffixes_to_remove)
    return name

def date_similarity(date1, date2):
    try:
        d1 = datetime.strptime(str(date1), '%d-%m-%Y')
        d2 = datetime.strptime(str(date2), '%d-%m-%Y')
        diff = abs((d1 - d2).days)
        return max(0, 1 - (diff / 30))  # Assume dates within 30 days could be a match
    except:
        return 0

def highlight_diff(s1, s2):
    d = difflib.Differ()
    diff = list(d.compare(s1, s2))
    result = []
    for i, s in enumerate(diff):
        if s[0] == ' ':
            result.append(s[-1])
        elif s[0] == '-':
            result.append(f'<span style="color: red;">{s[-1]}</span>')
    return ''.join(result)

def create_identifier(row, is_prj):
    if is_prj:
        return f"{row['SupplierGSTIN']}_{row['DocumentNumber']}"
    else:
        return f"{row['ctin']}_{row['inum']}"

def group_and_sum(data, is_prj):
    grouped = {}
    for _, row in data.iterrows():
        identifier = create_identifier(row, is_prj)
        if identifier not in grouped:
            grouped[identifier] = {
                'TaxableValue': 0,
                'TotalTax': 0,
                'TaxRate': 0
            }
        
        if is_prj:
            grouped[identifier]['TaxableValue'] += float(row['TaxableValue'])
            grouped[identifier]['TotalTax'] += float(row['IntegratedTaxAmount']) + float(row['CentralTaxAmount']) + float(row['StateUTTaxAmount'])
            grouped[identifier]['TaxRate'] = float(row['IntegratedTaxRate']) or (float(row['CentralTaxRate']) + float(row['StateUTTaxRate']))
        else:
            grouped[identifier]['TaxableValue'] += float(row['item_txval'])
            grouped[identifier]['TotalTax'] += float(row['item_igst']) + float(row['item_cgst']) + float(row['item_sgst'])
            grouped[identifier]['TaxRate'] = float(row['item_rt'])
    
    return grouped

def reconcile(prj_data, b2b_data):
    prj_grouped = group_and_sum(prj_data, True)
    b2b_grouped = group_and_sum(b2b_data, False)
    
    results = {}
    
    for identifier, prj_values in prj_grouped.items():
        if identifier in b2b_grouped:
            b2b_values = b2b_grouped[identifier]
            
            if abs(prj_values['TaxableValue'] - b2b_values['TaxableValue']) < 1 and abs(prj_values['TotalTax'] - b2b_values['TotalTax']) < 1:
                if abs(prj_values['TaxRate'] - b2b_values['TaxRate']) < 0.1:
                    results[identifier] = "Exact Match"
                else:
                    results[identifier] = "Partial Match - Tax Rate Mismatch"
            else:
                results[identifier] = "Partial Match - Amount Mismatch"
        else:
            results[identifier] = "No Match"
    
    for identifier in b2b_grouped:
        if identifier not in results:
            results[identifier] = "No Match in PRjswdppl.csv"
    
    return results

def perform_reconciliation(b2b_df, pr_df):
    # Initialize results
    reconciliation_results = {}
    match_counts = defaultdict(int)
    
    # Prepare dataframes
    try:
        b2b_df['identifier'] = b2b_df['ctin'].astype(str) + '_' + b2b_df['inum'].astype(str)
        pr_df['identifier'] = pr_df['SupplierGSTIN'].astype(str) + '_' + pr_df['DocumentNumber'].astype(str)
    except Exception as e:
        logger.error(f"Error creating identifiers: {str(e)}")
        return None

    # Group and sum B2B data
    try:
        b2b_grouped = b2b_df.groupby('identifier').agg({
            'ctin': 'first',
            'inum': 'first',
            'val': 'sum',
            'item_txval': 'sum',
            'item_igst': 'sum',
            'item_cgst': 'sum',
            'item_sgst': 'sum',
            'item_rt': 'first',
            'trdnm': 'first'  # Include customer name from B2B data
        }).reset_index()
        b2b_grouped['total_tax'] = b2b_grouped['item_igst'].astype(float) + b2b_grouped['item_cgst'].astype(float) + b2b_grouped['item_sgst'].astype(float)
    except Exception as e:
        logger.error(f"Error grouping B2B data: {str(e)}")
        return None

    # Group and sum PR data
    try:
        pr_grouped = pr_df.groupby('identifier').agg({
            'SupplierGSTIN': 'first',
            'DocumentNumber': 'first',
            'InvoiceValue': 'sum',
            'TaxableValue': 'sum',
            'IntegratedTaxAmount': 'sum',
            'CentralTaxAmount': 'sum',
            'StateUTTaxAmount': 'sum',
            'IntegratedTaxRate': 'first',
            'CentralTaxRate': 'first',
            'StateUTTaxRate': 'first',
            'SupplierName': 'first'  # Include customer name from PR data
        }).reset_index()
        pr_grouped['total_tax'] = pr_grouped['IntegratedTaxAmount'].astype(float) + pr_grouped['CentralTaxAmount'].astype(float) + pr_grouped['StateUTTaxAmount'].astype(float)
        pr_grouped['tax_rate'] = pr_grouped['IntegratedTaxRate'].astype(float) + pr_grouped['CentralTaxRate'].astype(float) + pr_grouped['StateUTTaxRate'].astype(float)
    except Exception as e:
        logger.error(f"Error grouping PR data: {str(e)}")
        return None

    # Perform reconciliation
    for _, pr_row in pr_grouped.iterrows():
        try:
            pr_identifier = pr_row['identifier']
            pr_gstin, pr_doc_num = pr_identifier.split('_')
            
            # Check if GSTIN exists in B2B data
            matching_b2b = b2b_grouped[b2b_grouped['ctin'] == pr_gstin]
            
            if matching_b2b.empty:
                status = "No Match - GSTIN not found in B2B data"
                reconciliation_results[pr_identifier] = {
                    'status': status,
                    'gstin_match': 'No Match',
                    'invoice_match': 'No Match',
                    'taxable_value_match': 'No Match',
                    'tax_match': 'No Match',
                    'tax_rate_match': 'No Match',
                    'b2b_customer_name': 'N/A',
                    'pr_customer_name': pr_row['SupplierName']
                }
            else:
                # Check if document number matches
                exact_match = matching_b2b[matching_b2b['inum'] == pr_doc_num]
                
                if not exact_match.empty:
                    b2b_row = exact_match.iloc[0]
                    
                    # Compare values
                    gstin_match = 'Match' if pr_gstin == b2b_row['ctin'] else 'No Match'
                    invoice_match = 'Match' if pr_doc_num == b2b_row['inum'] else 'No Match'
                    
                    invoice_diff = abs(float(pr_row['InvoiceValue']) - float(b2b_row['val']))
                    taxable_diff = abs(float(pr_row['TaxableValue']) - float(b2b_row['item_txval']))
                    tax_diff = abs(float(pr_row['total_tax']) - float(b2b_row['total_tax']))
                    tax_rate_diff = abs(float(pr_row['tax_rate']) - float(b2b_row['item_rt']))
                    
                    taxable_value_match = 'Match' if taxable_diff < 1 else 'Mismatch'
                    tax_match = 'Match' if tax_diff < 1 else 'Mismatch'
                    tax_rate_match = 'Match' if tax_rate_diff < 0.1 else 'Mismatch'
                    
                    if gstin_match == 'Match' and invoice_match == 'Match' and taxable_value_match == 'Match' and tax_match == 'Match' and tax_rate_match == 'Match':
                        status = "Exact Match"
                    else:
                        status = "Partial Match"
                    
                    reconciliation_results[pr_identifier] = {
                        'status': status,
                        'gstin_match': gstin_match,
                        'invoice_match': invoice_match,
                        'taxable_value_match': taxable_value_match,
                        'tax_match': tax_match,
                        'tax_rate_match': tax_rate_match,
                        'invoice_diff': invoice_diff,
                        'taxable_diff': taxable_diff,
                        'tax_diff': tax_diff,
                        'tax_rate_diff': tax_rate_diff,
                        'b2b_customer_name': b2b_row['trdnm'],
                        'pr_customer_name': pr_row['SupplierName']
                    }
                else:
                    status = "No Match - Document Number not found in B2B data"
                    reconciliation_results[pr_identifier] = {
                        'status': status,
                        'gstin_match': 'Match',
                        'invoice_match': 'No Match',
                        'taxable_value_match': 'No Match',
                        'tax_match': 'No Match',
                        'tax_rate_match': 'No Match',
                        'b2b_customer_name': matching_b2b.iloc[0]['trdnm'],
                        'pr_customer_name': pr_row['SupplierName']
                    }
            
            match_counts[status] += 1
        except Exception as e:
            logger.error(f"Error processing row: {pr_identifier}. Error: {str(e)}")
            status = "Error in processing"
            reconciliation_results[pr_identifier] = {
                'status': status,
                'error': str(e),
                'b2b_customer_name': 'Error',
                'pr_customer_name': 'Error'
            }
            match_counts[status] += 1

    # Check for B2B entries not in PR data
    for _, b2b_row in b2b_grouped.iterrows():
        if b2b_row['identifier'] not in reconciliation_results:
            status = "No Match - Not found in PR data"
            reconciliation_results[b2b_row['identifier']] = {
                'status': status,
                'gstin_match': 'No Match',
                'invoice_match': 'No Match',
                'taxable_value_match': 'No Match',
                'tax_match': 'No Match',
                'tax_rate_match': 'No Match',
                'b2b_customer_name': b2b_row['trdnm'],
                'pr_customer_name': 'N/A'
            }
            match_counts[status] += 1

    # Calculate percentages
    total_records = len(reconciliation_results)
    match_percentages = {k: (v / total_records) * 100 for k, v in match_counts.items()}

    # Add reconciliation results to dataframes
    b2b_grouped['reconciliation_status'] = b2b_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('status', 'Unknown'))
    b2b_grouped['gstin_match'] = b2b_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('gstin_match', 'Unknown'))
    b2b_grouped['invoice_match'] = b2b_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('invoice_match', 'Unknown'))
    b2b_grouped['taxable_value_match'] = b2b_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('taxable_value_match', 'Unknown'))
    b2b_grouped['tax_match'] = b2b_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('tax_match', 'Unknown'))
    b2b_grouped['tax_rate_match'] = b2b_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('tax_rate_match', 'Unknown'))

    pr_grouped['reconciliation_status'] = pr_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('status', 'Unknown'))
    pr_grouped['gstin_match'] = pr_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('gstin_match', 'Unknown'))
    pr_grouped['invoice_match'] = pr_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('invoice_match', 'Unknown'))
    pr_grouped['taxable_value_match'] = pr_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('taxable_value_match', 'Unknown'))
    pr_grouped['tax_match'] = pr_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('tax_match', 'Unknown'))
    pr_grouped['tax_rate_match'] = pr_grouped['identifier'].map(lambda x: reconciliation_results.get(x, {}).get('tax_rate_match', 'Unknown'))

    return {
        'b2b_df': b2b_grouped,
        'pr_df': pr_grouped,
        'reconciliation_results': reconciliation_results,
        'match_counts': dict(match_counts),
        'match_percentages': match_percentages,
        'overall_percentage': (match_counts.get("Exact Match", 0) / total_records) * 100 if total_records > 0 else 0
    }
    
def combine_json_files(uploaded_jsons):
    combined_data = {'data': {'docdata': {'b2b': []}}}
    for uploaded_json in uploaded_jsons:
        data = json.load(uploaded_json)
        combined_data['data']['docdata']['b2b'].extend(data['data']['docdata']['b2b'])
    return combined_data


def create_enhanced_summary(b2b_df, pr_df, reconciliation_results):
    # Create a combined dataframe
    combined_df = pd.merge(
        b2b_df, pr_df,
        left_on=['ctin', 'inum'],
        right_on=['SupplierGSTIN', 'DocumentNumber'],
        how='outer',
        suffixes=('_b2b', '_pr')
    )

    # Group by GST Number and sum the amounts
    summary = combined_df.groupby('ctin').agg({
        'trdnm': 'first',
        'SupplierName': 'first',
        'val': 'sum',
        'InvoiceValue': 'sum',
        'item_igst': 'sum',
        'item_cgst': 'sum',
        'item_sgst': 'sum',
        'IntegratedTaxAmount': 'sum',
        'CentralTaxAmount': 'sum',
        'StateUTTaxAmount': 'sum',
    }).reset_index()

    # Rename columns for clarity
    summary.columns = [
        'GST Number', 'B2B Customer Name', 'PR Customer Name',
        'B2B Invoice Amount', 'PR Invoice Amount',
        'B2B IGST', 'B2B CGST', 'B2B SGST',
        'PR IGST', 'PR CGST', 'PR SGST'
    ]

    # Calculate total tax amounts
    summary['B2B Tax Amount'] = summary['B2B IGST'] + summary['B2B CGST'] + summary['B2B SGST']
    summary['PR Tax Amount'] = summary['PR IGST'] + summary['PR CGST'] + summary['PR SGST']

    # Calculate excess amounts
    summary['Excess in B2B Invoice Amount'] = summary.apply(
        lambda row: row['B2B Invoice Amount'] - row['PR Invoice Amount'] if row['B2B Invoice Amount'] > row['PR Invoice Amount'] else 0,
        axis=1
    )

    summary['Excess in PR Invoice Amount'] = summary.apply(
        lambda row: row['PR Invoice Amount'] - row['B2B Invoice Amount'] if row['PR Invoice Amount'] > row['B2B Invoice Amount'] else 0,
        axis=1
    )

    summary['Excess in B2B Tax Amount'] = summary.apply(
        lambda row: row['B2B Tax Amount'] - row['PR Tax Amount'] if row['B2B Tax Amount'] > row['PR Tax Amount'] else 0,
        axis=1
    )

    summary['Excess in PR Tax Amount'] = summary.apply(
        lambda row: row['PR Tax Amount'] - row['B2B Tax Amount'] if row['PR Tax Amount'] > row['B2B Tax Amount'] else 0,
        axis=1
    )

    # Replace inf and -inf with NaN and then with 0
    summary = summary.replace([np.inf, -np.inf, np.nan], 0)

    # Determine overall status for each GST Number
    def determine_status(row):
        if pd.isna(row['B2B Customer Name']):
            return 'GSTIN Not Found in B2B'
        elif pd.isna(row['PR Customer Name']):
            return 'GSTIN Not Found in PR'
        else:
            invoice_excess_b2b = row['Excess in B2B Invoice Amount'] < 1
            invoice_excess_pr = row['Excess in PR Invoice Amount'] < 1
            tax_excess_b2b = row['Excess in B2B Tax Amount'] < 1
            tax_excess_pr = row['Excess in PR Tax Amount'] < 1

            invoice_match = invoice_excess_b2b and invoice_excess_pr
            tax_match = tax_excess_b2b and tax_excess_pr

            if invoice_match and tax_match:
                return 'Exact Match'
            elif invoice_match or tax_match:
                return 'Partial Match'
            else:
                return 'No Match'

    summary['Status'] = summary.apply(determine_status, axis=1)

    # Move GSTIN not found statuses to the bottom of the summary
    status_order = {'Exact Match': 1, 'Partial Match': 2, 'No Match': 3, 'GSTIN Not Found in B2B': 4, 'GSTIN Not Found in PR': 5}
    summary['status_order'] = summary['Status'].map(status_order)
    summary = summary.sort_values('status_order')

    # Drop the temporary 'status_order' column
    summary = summary.drop(columns=['status_order'])

    # Reorder columns
    column_order = [
        'GST Number', 'B2B Customer Name', 'PR Customer Name',
        'B2B Invoice Amount', 'PR Invoice Amount',
        'Excess in B2B Invoice Amount', 'Excess in PR Invoice Amount',
        'B2B Tax Amount', 'PR Tax Amount',
        'Excess in B2B Tax Amount', 'Excess in PR Tax Amount',
        'Status'
    ]
    summary = summary[column_order]

    return summary


# Add this function to handle the SupplierGSTIN issue
def clean_supplier_gstin(df):
    if 'SupplierGSTIN' in df.columns:
        df['SupplierGSTIN'] = df['SupplierGSTIN'].astype(str)
    return df


def copy_to_clipboard(text):
    pyperclip.copy(text)
    st.success(f"Copied '{text}' to clipboard!")

# Function to show required columns with copy-to-clipboard buttons
# Function to show required columns with copy-to-clipboard buttons and explanations
def show_pr_info_expander():
    with st.expander("Purchase Register Format Info - Required Columns"):
        st.write("For the reconciliation process to work correctly, the Purchase Register (PR) file must contain the following mandatory columns. Each column serves a specific purpose in the process:")

        # List of required columns and explanations
        required_columns = [
            ('SupplierGSTIN', 'The GST Identification Number (GSTIN) of the supplier. It is essential for matching the PR entries with the corresponding B2B data.'),
            ('DocumentNumber', 'The invoice number from the supplier. This is critical for identifying and matching specific invoices between PR and B2B data.'),
            ('DocumentDate', 'The date of the invoice. This helps verify the accuracy of transactions and identify any discrepancies in timing.'),
            ('InvoiceValue', 'The total value of the invoice, including tax. Used to reconcile the overall invoice value between PR and B2B data.'),
            ('TaxableValue', 'The taxable value of the goods or services. This is necessary for ensuring the correct tax base is being reconciled.'),
            ('IntegratedTaxAmount', 'The amount of Integrated GST (IGST) charged on the invoice. This is used to reconcile tax amounts between PR and B2B data.'),
            ('CentralTaxAmount', 'The amount of Central GST (CGST) charged on the invoice. This is used to reconcile tax amounts between PR and B2B data.'),
            ('StateUTTaxAmount', 'The amount of State or Union Territory GST (SGST/UTGST) charged on the invoice. This is used to reconcile tax amounts between PR and B2B data.'),
            ('IntegratedTaxRate', 'The tax rate applied for IGST on the invoice. This helps verify whether the correct tax rate was applied.'),
            ('CentralTaxRate', 'The tax rate applied for CGST on the invoice. This helps verify whether the correct tax rate was applied.'),
            ('StateUTTaxRate', 'The tax rate applied for SGST/UTGST on the invoice. This helps verify whether the correct tax rate was applied.'),
            ('SupplierName', 'The name of the supplier. This column is used for reference and to ensure clarity in identifying suppliers during the reconciliation process.')
        ]

        # Display each column with explanation and copy-to-clipboard button
        for column, description in required_columns:
            col1, col2, col3 = st.columns([3, 4, 3])
            with col1:
                st.write(f"**{column}**")
            with col2:
                st.write(description)
            with col3:
                if st.button(f"Copy {column}", key=f"copy_{column}"):
                    copy_to_clipboard(column)


def main():
    st.title("Enhanced B2B and PR Data Reconciliation")

    # Info expander for PR format
    show_pr_info_expander()

    uploaded_jsons = st.file_uploader("Choose JSON files", type="json", accept_multiple_files=True, key="json_uploader")

    if uploaded_jsons:
        try:
            # Combine JSON files
            combined_json_data = combine_json_files(uploaded_jsons)
            
            # Process combined B2B data
            b2b_data = combined_json_data['data']['docdata']['b2b']
            processed_data = process_b2b_data(b2b_data)
            b2b_df = pd.DataFrame(processed_data)
            
            st.subheader("Combined B2B Data")
            st.dataframe(b2b_df)

            uploaded_pr = st.file_uploader("Choose PR file (CSV or Excel)", type=["csv", "xlsx"], key="pr_uploader")
            
            if uploaded_pr is not None:
                try:
                    if uploaded_pr.name.endswith('.csv'):
                        pr_df = pd.read_csv(uploaded_pr)
                    else:
                        pr_df = pd.read_excel(uploaded_pr)
                    
                    # Clean SupplierGSTIN column
                    pr_df = clean_supplier_gstin(pr_df)
                    
                    st.subheader("PR Data")
                    st.dataframe(pr_df)
                    
                    # Clean and prepare data
                    b2b_df = clean_and_prepare_data(b2b_df)
                    pr_df = clean_and_prepare_data(pr_df)
                    
                    # Ensure date columns are in the correct format
                    b2b_df['dt'] = pd.to_datetime(b2b_df['dt'], format='%d-%m-%Y', errors='coerce')
                    pr_df['DocumentDate'] = pd.to_datetime(pr_df['DocumentDate'], errors='coerce')
                    
                    # Perform reconciliation
                    results = perform_reconciliation(b2b_df, pr_df)
                    
                    if results is None:
                        st.error("An error occurred during reconciliation. Please check the logs for more information.")
                        return

                    try:
                        summary = create_enhanced_summary(results['b2b_df'], results['pr_df'], results['reconciliation_results'])
                    except Exception as e:
                        st.error(f"An error occurred while creating the summary: {str(e)}")
                        return

                    # Generate report
                    st.subheader("Reconciliation Report")
                    total_records = len(results['b2b_df'])
                    st.write(f"Total records: {total_records}")
                    for status, count in results['match_counts'].items():
                        st.write(f"{status}: {count} ({results['match_percentages'][status]:.2f}%)")
                    st.write(f"Overall Exact Match Percentage: {results['overall_percentage']:.2f}%")

                    # Display detailed match statuses with colors for B2B
                    st.subheader("Detailed Match Statuses - B2B Data")
                    
                    def color_match_status(val):
                        if val == 'Match':
                            return 'background-color: green'
                        elif val == 'Mismatch' or val == 'No Match':
                            return 'background-color: red'
                        else:
                            return ''
                    
                    styled_b2b_df = results['b2b_df'].style.applymap(color_match_status, subset=['gstin_match', 'invoice_match', 'taxable_value_match', 'tax_match', 'tax_rate_match'])
                    st.dataframe(styled_b2b_df)

                    # Display detailed match statuses with colors for PR
                    st.subheader("Detailed Match Statuses - PR Data")
                    styled_pr_df = results['pr_df'].style.applymap(color_match_status, subset=['gstin_match', 'invoice_match', 'taxable_value_match', 'tax_match', 'tax_rate_match'])
                    st.dataframe(styled_pr_df)

                    # Option to view detailed match information
                    st.subheader("View Detailed Match Information")
                    selected_invoice = st.selectbox("Select an invoice to view detailed match information:", results['b2b_df']['inum'])
                    if selected_invoice:
                        b2b_details = results['b2b_df'][results['b2b_df']['inum'] == selected_invoice].iloc[0]
                        pr_details = results['pr_df'][results['pr_df']['DocumentNumber'] == selected_invoice]
                        
                        st.write("Match Details:")
                        st.write(f"Reconciliation Status: {b2b_details['reconciliation_status']}")
                        st.write(f"Invoice Number: {b2b_details['inum']}")
                        st.write(f"Supplier GSTIN: B2B: {b2b_details['ctin']}")
                        
                        if not pr_details.empty:
                            pr_row = pr_details.iloc[0]
                            st.write(f"Supplier GSTIN: PR: {pr_row['SupplierGSTIN']}")
                            st.write(f"GSTIN Match: {b2b_details['gstin_match']}")
                            st.write(f"Invoice Match: {b2b_details['invoice_match']}")
                            st.write(f"Invoice Value: B2B: {b2b_details['val']}, PR: {pr_row['InvoiceValue']}")
                            st.write(f"Taxable Value: B2B: {b2b_details['item_txval']}, PR: {pr_row['TaxableValue']}")
                            st.write(f"Taxable Value Match: {b2b_details['taxable_value_match']}")
                            b2b_total_tax = b2b_details['item_igst'] + b2b_details['item_cgst'] + b2b_details['item_sgst']
                            pr_total_tax = pr_row['IntegratedTaxAmount'] + pr_row['CentralTaxAmount'] + pr_row['StateUTTaxAmount']
                            st.write(f"Total Tax: B2B: {b2b_total_tax}, PR: {pr_total_tax}")
                            st.write(f"Tax Match: {b2b_details['tax_match']}")
                            st.write(f"Tax Rate: B2B: {b2b_details['item_rt']}, PR: {pr_row['tax_rate']}")
                            st.write(f"Tax Rate Match: {b2b_details['tax_rate_match']}")
                        else:
                            st.write("No matching PR data found for this invoice.")

                    # Enhanced Combined Reconciliation Summary
                    st.subheader("Enhanced Combined Reconciliation Summary")
    
                    # Apply conditional formatting
                    def color_excess(val):
                        color = 'red' if val > 0 else 'black'
                        return f'color: {color}'
    
                    def color_status(val):
                        if val == 'Exact Match':
                            return 'background-color: green; color: white'
                        elif val == 'Partial Match':
                            return 'background-color: yellow'
                        else:
                            return 'background-color: red; color: white'
    
                    styled_summary = summary.style\
                        .applymap(color_excess, subset=[
                            'Excess in B2B Invoice Amount', 'Excess in PR Invoice Amount',
                            'Excess in B2B Tax Amount', 'Excess in PR Tax Amount'
                        ])\
                        .applymap(color_status, subset=['Status'])\
                        .format({
                            'B2B Invoice Amount': '{:.2f}',
                            'PR Invoice Amount': '{:.2f}',
                            'Excess in B2B Invoice Amount': '{:.2f}',
                            'Excess in PR Invoice Amount': '{:.2f}',
                            'B2B Tax Amount': '{:.2f}',
                            'PR Tax Amount': '{:.2f}',
                            'Excess in B2B Tax Amount': '{:.2f}',
                            'Excess in PR Tax Amount': '{:.2f}'
                        })
    
                    st.dataframe(styled_summary)
    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    total_invoices = len(summary)
                    exact_matches = (summary['Status'] == 'Exact Match').sum()
                    partial_matches = (summary['Status'] == 'Partial Match').sum()
                    no_matches = total_invoices - exact_matches - partial_matches
    
                    st.write(f"Total Invoices: {total_invoices}")
                    st.write(f"Exact Matches: {exact_matches} ({exact_matches/total_invoices*100:.2f}%)")
                    st.write(f"Partial Matches: {partial_matches} ({partial_matches/total_invoices*100:.2f}%)")
                    st.write(f"No Matches: {no_matches} ({no_matches/total_invoices*100:.2f}%)")
    
                    # Create Excel file with multiple sheets
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        results['b2b_df'].to_excel(writer, sheet_name='B2B Data', index=False)
                        results['pr_df'].to_excel(writer, sheet_name='PR Data', index=False)
                        summary.to_excel(writer, sheet_name='Reconciliation Summary', index=False)
                    
                    st.download_button(
                        label="Download Reconciled Data",
                        data=output.getvalue(),
                        file_name="reconciled_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                except Exception as e:
                    st.error(f"An error occurred while processing the PR file: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred while processing the JSON files: {str(e)}")

if __name__ == "__main__":
    main()