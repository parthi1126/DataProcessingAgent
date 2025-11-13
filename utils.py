# utils.py
# Our finalized, debugged data analysis library.

import pandas as pd
import numpy as np
from scipy import stats
import re
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
import warnings

# --- 1. SETUP AND LLM INITIALIZATION ---

# Load variables from .env file (like GOOGLE_API_KEY)
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", message="Could not infer format")
# Suppress the Google API core warning
warnings.filterwarnings("ignore", category=FutureWarning, module='google.api_core._python_version_support')


# LLM Setup
# Load the API key from the environment variables.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = None

if GOOGLE_API_KEY:
    try:
        # Using the correct 1.5 Pro model from your project plan
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", 
            google_api_key=GOOGLE_API_KEY, 
            temperature=0.1
        )
        print("utils.py: LLM initialized successfully.")
    except Exception as e:
        print(f"utils.py: LLM init warning: {e}. Will use mock reports.")
else:
    print("utils.py: WARNING: GOOGLE_API_KEY not found in .env file. Will use mock reports.")


# --- 2. ALL DATA ANALYSIS FUNCTIONS ---

def parse_indian_number(text: str) -> float:
    """Convert Indian number formats (e.g., '1.2 Lakh', '10Cr', '10,000') to float."""
    if not isinstance(text, str) or text.lower() in ['na', 'n/a', '']:
        return np.nan
    
    text = text.lower().strip()
    text = re.sub(r'[₹$,]', '', text)  # Remove currency symbols
    text = re.sub(r'[,]', '', text)  # Remove thousand separators
    
    # This is the critical data parsing fix
    text = re.sub(r'per sqft', '', text, flags=re.IGNORECASE)  
    
    multipliers = {'lakh': 100000, 'l': 100000, 'crore': 10000000, 'cr': 10000000, 'k': 1000}
    for unit, multiplier in multipliers.items():
        if unit in text:
            try:
                # Remove all non-numeric characters except the decimal point
                number_str = re.sub(r'[^0-9.]', '', text)
                if number_str:
                    return float(number_str) * multiplier
                else:
                    return np.nan
            except:
                return np.nan
    try:
        # Handle simple numbers like '2891'
        return float(text)
    except:
        return np.nan

def ingest_data(file_path: str, max_size_mb: int = 50) -> dict:
    """
    Ingest CSV/Excel/JSON. Returns dict with data preview, metadata, and sample.
    Robust cleaning for numeric/date columns with Indian number formats.
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"File too large ({file_size_mb:.1f} MB). Use Dask/PySpark for bigger files.")

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type: Use CSV, Excel, or JSON.")

    # FIX for 'square_feet' bug: Smarter parsing logic
    for col in df.columns:
        if df[col].dtype == 'object':
            col_low = col.lower()
            
            # Use special parser ONLY for price/cost columns
            if 'price' in col_low or 'cost' in col_low:
                df[col] = df[col].apply(parse_indian_number)
            
            # Use simple numeric parser for 'sqft'
            elif 'sqft' in col_low or 'square' in col_low or 'feet' in col_low:
                # This will correctly parse '1650.0' and ignore 'nan'
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Parse dates
            elif 'date' in col_low:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    # Preview
    preview = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'head': df.head().to_dict('records'),
        'missing': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'size_mb': file_size_mb,
        'sample_rows': df.sample(5, random_state=42).to_dict('records')  # Random 5 rows, fixed seed
    }
    return {'data': df, 'preview': preview}

def generate_summary_report(preview: dict) -> str:
    """Simple natural language summary (pre-LLM)."""
    shape = preview['shape']
    missing_count = sum(preview['missing'].values())
    return f"Dataset: {shape[0]} rows x {shape[1]} columns. Missing values: {missing_count}. Duplicates: {preview['duplicates']}."

def analyze_data(df: pd.DataFrame, file_path: str = None) -> Dict[str, Any]:
    """
    Comprehensive data analysis with all summaries. Returns dict for LLM consumption.
    """
    analysis = {}
    
    # A. General Information
    analysis['general'] = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage_bytes': df.memory_usage(deep=True).sum(),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024**2), 2),
    }
    if file_path:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        analysis['general']['file_size_mb'] = round(file_size_mb, 2)
        if file_size_mb < 50:
            analysis['general']['recommended_framework'] = 'Pandas'
        elif file_size_mb < 2000:
            analysis['general']['recommended_framework'] = 'Dask'
        else:
            analysis['general']['recommended_framework'] = 'PySpark/BigQuery'
    
    # B. Missing Values & Nulls
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    analysis['missing'] = {
        'count_per_col': missing_count.to_dict(),
        'pct_per_col': missing_pct.to_dict(),
        'completely_null_cols': [col for col, cnt in missing_count.items() if cnt == len(df)],
        'total_missing': int(missing_count.sum()),
        'total_missing_pct': round(missing_pct.mean(), 2)
    }
    
    # C. Duplicate Detection
    dup_rows = df.duplicated().sum()
    analysis['duplicates'] = {
        'row_count': int(dup_rows),
        'dup_pct': round(dup_rows / len(df) * 100, 2) if len(df) > 0 else 0,
        'col_duplicates': []
    }
    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            if (df.iloc[:, i] == df.iloc[:, j]).all():
                analysis['duplicates']['col_duplicates'].append((df.columns[i], df.columns[j]))
    
    # D. Summary Statistics (Numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        desc = df[numeric_cols].describe().round(2).to_dict()
        q1 = df[numeric_cols].quantile(0.25)
        q3 = df[numeric_cols].quantile(0.75)
        iqr = q3 - q1
        analysis['numeric_summary'] = {
            'describe': desc,
            'iqr_per_col': iqr.to_dict(),
            'skewness': df[numeric_cols].skew().round(2).to_dict(),
            'kurtosis': df[numeric_cols].kurtosis().round(2).to_dict()
        }
    else:
        analysis['numeric_summary'] = {'message': 'No numeric columns found'}
    
    # E. Summary Statistics (Categorical)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    analysis['categorical_summary'] = {}
    for col in cat_cols:
        unique = df[col].nunique()
        top5 = df[col].value_counts().head(5).to_dict() if not df[col].isnull().all() else {}
        analysis['categorical_summary'][col] = {
            'unique_count': unique,
            'cardinality_high': unique > len(df) / 10,
            'top_values': top5
        }
    
    # F. Feature Correlation (Numeric only)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().round(2)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:
                    high_corr_pairs.append({
                        'cols': [corr_matrix.columns[i], corr_matrix.columns[j]],
                        'corr': round(corr_matrix.iloc[i, j], 2)
                    })
        analysis['correlation'] = {
            'matrix': corr_matrix.to_dict(),
            'high_corr_pairs': high_corr_pairs,
            'recommendations': ['Drop or aggregate highly correlated features' if high_corr_pairs else 'No major multicollinearity detected']
        }
    else:
        analysis['correlation'] = {
            'message': 'Insufficient numeric columns for correlation',
            'recommendations': ['Convert object columns to numeric where appropriate for correlation analysis']
        }
    
    # G. Outlier Detection (IQR method)
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        out_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers[col] = {
            'count': int(out_count),
            'pct': round(out_count / len(df) * 100, 2),
            'bounds': {'lower': round(lower_bound, 2), 'upper': round(upper_bound, 2)}
        }
    analysis['outliers'] = outliers
    analysis['outlier_method'] = 'IQR (1.5 * IQR rule)'
    
    # H. Domain-Specific Features
    domain_flags = {
        'potential_id_cols': [],
        'potential_date_cols': [],
        'potential_monetary_cols': [],
        'high_missing_cols': [col for col, pct in analysis['missing']['pct_per_col'].items() if pct > 50],
        'high_cardinality_cols': [col for col, data in analysis['categorical_summary'].items() if data['cardinality_high']]
    }
    for col in df.columns:
        # ID: unique, int-like or starts with 'id'
        if df[col].nunique() == len(df) and ('id' in col.lower() or df[col].dtype in ['int64', 'object']):
            domain_flags['potential_id_cols'].append(col)
        
        # Date: Only try likely columns
        if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                pd.to_datetime(df[col].dropna(), errors='raise')
                domain_flags['potential_date_cols'].append(col)
            except:
                pass
        
        # Monetary: col name hints, positive floats
        if any(kw in col.lower() for kw in ['amount', 'price', 'cost', 'revenue']) and df[col].dtype in ['float64']:
            pos_vals = df[col].dropna()
            if len(pos_vals) > 0 and (pos_vals > 0).all():
                domain_flags['potential_monetary_cols'].append(col)
    
    # Infer cat vs num
    for col in df.columns:
        inferred_type = 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
        domain_flags[f'{col}_inferred_type'] = inferred_type
    
    analysis['domain_flags'] = domain_flags
    
    # I. Quick Visual Insights (Text summaries)
    analysis['visual_insights'] = {
        'numeric_hist_summary': {},
        'cat_count_summary': {},
        'scatter_summary': 'TBD (pairplot for top 3 num cols if >2)'
    }
    for col in numeric_cols[:3]:
        if len(df[col].dropna()) > 0:
            analysis['visual_insights']['numeric_hist_summary'][col] = f"Distribution: mean {df[col].mean():.2f}, std {df[col].std():.2f}"
    for col in cat_cols[:3]:
        if len(df[col].dropna()) > 0:
            top_val = df[col].value_counts().index[0] if not df[col].value_counts().empty else 'N/A'
            analysis['visual_insights']['cat_count_summary'][col] = f"Most frequent: {top_val}"
    
    return analysis

def generate_quality_report(preview: dict, analysis: dict, llm: Optional[ChatGoogleGenerativeAI] = None) -> str:
    """LLM-powered report based on full analysis. Uses provided llm or global/mock."""
    current_llm = llm or globals().get('llm')
    
    if current_llm is None:
        # Fallback mock report if LLM fails to initialize
        issues = []
        if analysis['missing']['total_missing'] > 0:
            issues.append(f"Missing values: {analysis['missing']['total_missing']} total ({analysis['missing']['total_missing_pct']}%)")
        if analysis['duplicates']['row_count'] > 0:
            issues.append(f"Duplicates: {analysis['duplicates']['row_count']} rows")
        if analysis['numeric_summary'].get('message'):
            issues.append("No numeric columns; suggest converting 'price', 'square_feet' to float")
        recs = "Impute missing with mode/median; drop duplicates; parse strings to numerics for 'price'/'square_feet'."
        return f"MOCK REPORT: Overview - {analysis['general']['shape']} dataset with {len(analysis['general']['columns'])} features.\nKey Issues:\n- {'; '.join(issues)}\nInsights & Recommendations:\n- {recs}\nNext Steps: Clean data and proceed to feature engineering."
    
    prompt = f"""
    Generate a concise, professional data quality report (300-500 words) for this dataset.
    Structure: Overview | Key Issues (missing/duplicates/outliers/corrs) | Insights & Recommendations | Next Steps.
    Use bullet points where helpful. Be actionable (e.g., 'Impute with median for col X').
    
    Preview: {preview}
    Full Analysis: {analysis}
    """
    try:
        response = current_llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"LLM Error: {str(e)}. Mock report: Dataset has {analysis['general']['shape'][0]} rows, {analysis['general']['shape'][1]} cols. Check analysis for details."

def combine_analysis_outputs(file_path: str) -> Dict[str, Any]:
    """
    Combines all outputs: ingestion, analysis, summary, quality report, and sample rows.
    Returns a single dict with all results.
    """
    result = {} # Start with a clean dict
    
    # Ingest
    try:
        ingestion_result = ingest_data(file_path)
        result['ingestion'] = ingestion_result['preview'] 
        df = ingestion_result['data']
    except Exception as e:
        result['error'] = f"Ingestion failed: {str(e)}"
        return result
    
    # Simple Summary
    result['summary'] = generate_summary_report(result['ingestion'])
    
    # Full Analysis
    try:
        result['analysis'] = analyze_data(df, file_path=file_path)
    except Exception as e:
        result['error'] = f"Analysis failed: {str(e)}"
        return result
    
    # Quality Report
    try:
        result['quality_report'] = generate_quality_report(result['ingestion'], result['analysis'], llm=llm)
    except Exception as e:
        result['quality_report'] = f"Report failed: {str(e)}"
    
    # Sample Rows
    result['sample_rows'] = result['ingestion']['sample_rows']
    
    # --- THIS IS THE ONLY CHANGE ---
    # Add the loaded dataframe to the result dict
    result['dataframe'] = df
    
    return result