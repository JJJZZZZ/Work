from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import json
import base64
from typing import List, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import your custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our refactored modules
from core.data_processor import DataProcessor
from core.visualizations import MatchingVisualizer

try:
    from mongocausalml.matchingtool.data_validator_propensity import DataFrameValidator
    import mongocausalml.matchingtool.utils as utils
    import mongocausalml.matchingtool.helper as helper
    import mongocausalml.matchingtool.propensity_matcher as propensity_matcher
    MONGODB_AVAILABLE = True
except ImportError:
    # Fallback for development - create mock classes
    print("MongoDB packages not available. Using fallback implementations.")
    MONGODB_AVAILABLE = False
    
    class MockDataFrameValidator:
        def __init__(self, df, id_column, assignment_column):
            self.df = df
            self.id_column = id_column
            self.assignment_column = assignment_column
        
        def validate(self):
            return {"validation": "passed"}
    
    class MockPropensityScoreMatching:
        def __init__(self, **kwargs):
            self.config = kwargs
        
        def get_list_month(self, df):
            return df[self.config['ds_month_column']].unique().tolist()
        
        def get_feature_type(self, df):
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if 'treatment_flag' in numeric_cols:
                numeric_cols.remove('treatment_flag')
            string_cols = df.select_dtypes(include=['object']).columns.tolist()
            return numeric_cols, string_cols
        
        def popensity_matching_monthly(self, df, month, string_columns, numeric_columns, target_columns):
            # Simple mock matching - just return a subset of the data
            return df.sample(frac=0.7).reset_index(drop=True)
    
    DataFrameValidator = MockDataFrameValidator
    propensity_matcher = type('MockModule', (), {
        'PropensityScoreMatching': MockPropensityScoreMatching
    })()
    
    helper = type('MockHelper', (), {
        'keep_columns_and_create_treatment_flag_propensity': lambda df, id_col, match_cols, date_col, treat_col: df,
        'winsorize_iqr': lambda df, numeric_columns: df
    })()

app = FastAPI(title="Propensity Matching Tool", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variable to store data between requests
app.state.data_store = {}

# Initialize processors
data_processor = DataProcessor()
visualizer = MatchingVisualizer()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload and process CSV data"""
    try:
        print(f"DEBUG: Starting file upload for {file.filename}")
        
        # Read the uploaded file
        contents = await file.read()
        print(f"DEBUG: File read successfully, size: {len(contents)} bytes")
        
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        print(f"DEBUG: CSV parsed successfully, shape: {df.shape}")
        
        # Store data in app state
        app.state.data_store['original_data'] = df
        print(f"DEBUG: Data stored in app state")
        
        # Get comprehensive data summary
        print(f"DEBUG: Starting data summary generation...")
        summary = data_processor.get_data_summary(df)
        print(f"DEBUG: Data summary generated successfully")
        
        # Add ID columns detection
        id_columns = [col for col in df.columns if "id" in col.lower()]
        summary['id_columns'] = id_columns
        summary['sample_data'] = df.head().to_dict('records')
        print(f"DEBUG: Additional metadata added")
        
        print(f"DEBUG: Returning success response")
        return JSONResponse(content={"success": True, "summary": summary})
        
    except Exception as e:
        print(f"DEBUG: Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/configure-matching")
async def configure_matching(
    primary_identifier: str = Form(...),
    columns_for_matching: str = Form(...),
    ds_month_column: str = Form(...),
    treatment_flag: str = Form(...),
    handle_missing: str = Form(...),
    standardize_numeric: bool = Form(...),
    matching_strictness: float = Form(0.1),
    with_replacement: bool = Form(False)
):
    """Configure matching parameters and prepare data"""
    try:
        print(f"DEBUG: Received parameters:")
        print(f"  primary_identifier: {primary_identifier}")
        print(f"  columns_for_matching: {columns_for_matching}")
        print(f"  ds_month_column: {ds_month_column}")
        print(f"  treatment_flag: {treatment_flag}")
        print(f"  handle_missing: {handle_missing}")
        print(f"  standardize_numeric: {standardize_numeric}")
        
        if 'original_data' not in app.state.data_store:
            raise HTTPException(status_code=400, detail="No data uploaded")
        
        df = app.state.data_store['original_data'].copy()
        print(f"DEBUG: Original data shape: {df.shape}")
        print(f"DEBUG: Original data columns: {df.columns.tolist()}")
        
        # Parse columns for matching
        columns_for_matching_list = [col.strip() for col in columns_for_matching.split(',')]
        print(f"DEBUG: Parsed matching columns: {columns_for_matching_list}")
        
        # Prepare data using our data processor
        try:
            df_selected = data_processor.prepare_data_for_matching(
                df,
                primary_identifier,
                columns_for_matching_list,
                ds_month_column,
                treatment_flag,
            )
            print(f"DEBUG: Data preparation successful, shape: {df_selected.shape}")
        except Exception as prep_error:
            print(f"DEBUG: Data preparation failed: {str(prep_error)}")
            raise HTTPException(status_code=400, detail=f"Data preparation failed: {str(prep_error)}")
        
        # Handle missing values
        try:
            df_selected = data_processor.handle_missing_values(
                df_selected, 
                method=handle_missing,
                exclude_columns=[primary_identifier, ds_month_column]
            )
            print(f"DEBUG: Missing values handled, shape: {df_selected.shape}")
        except Exception as missing_error:
            print(f"DEBUG: Missing value handling failed: {str(missing_error)}")
            raise HTTPException(status_code=400, detail=f"Missing value handling failed: {str(missing_error)}")
        
        # Handle outliers using our data processor
        try:
            numeric_columns, string_columns = data_processor.get_column_types(df_selected)
            print(f"DEBUG: Column types - numeric: {numeric_columns}, string: {string_columns}")
            
            df_selected = data_processor.handle_outliers(df_selected, numeric_columns=numeric_columns)
            print(f"DEBUG: Outliers handled, shape: {df_selected.shape}")
        except Exception as outlier_error:
            print(f"DEBUG: Outlier handling failed: {str(outlier_error)}")
            raise HTTPException(status_code=400, detail=f"Outlier handling failed: {str(outlier_error)}")
        
        # Standardize numeric columns if requested
        if standardize_numeric and numeric_columns:
            try:
                df_selected, scaler = data_processor.standardize_numeric_columns(df_selected, numeric_columns)
                app.state.data_store['scaler'] = scaler
                print(f"DEBUG: Standardization completed")
            except Exception as std_error:
                print(f"DEBUG: Standardization failed: {str(std_error)}")
                raise HTTPException(status_code=400, detail=f"Standardization failed: {str(std_error)}")
        
        # Store processed data
        app.state.data_store['processed_data'] = df_selected
        app.state.data_store['config'] = {
            'primary_identifier': primary_identifier,
            'columns_for_matching': columns_for_matching_list,
            'ds_month_column': ds_month_column,
            'treatment_flag': treatment_flag,
            'matching_strictness': matching_strictness,
            'with_replacement': with_replacement
        }
        
        # Validate data
        try:
            validator = DataFrameValidator(
                df_selected, id_column=primary_identifier, assignment_column="treatment_flag"
            )
            validation_result = validator.validate()
            print(f"DEBUG: Validation completed")
        except Exception as val_error:
            print(f"DEBUG: Validation failed: {str(val_error)}")
            raise HTTPException(status_code=400, detail=f"Validation failed: {str(val_error)}")
        
        # Calculate group sizes
        try:
            treatment_size = df_selected[df_selected["treatment_flag"] == 1][primary_identifier].nunique()
            control_size = df_selected[df_selected["treatment_flag"] == 0][primary_identifier].nunique()
            print(f"DEBUG: Group sizes - treatment: {treatment_size}, control: {control_size}")
        except Exception as size_error:
            print(f"DEBUG: Group size calculation failed: {str(size_error)}")
            raise HTTPException(status_code=400, detail=f"Group size calculation failed: {str(size_error)}")
        
        return JSONResponse(content={
            "success": True,
            "validation": "passed",
            "treatment_size": treatment_size,
            "control_size": control_size,
            "processed_shape": df_selected.shape
        })
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"DEBUG: Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error configuring matching: {str(e)}")

@app.post("/run-matching")
async def run_matching():
    """Execute the propensity score matching"""
    try:
        if 'processed_data' not in app.state.data_store:
            raise HTTPException(status_code=400, detail="No processed data available")
        
        df_selected = app.state.data_store['processed_data']
        config = app.state.data_store['config']
        
        # Initialize the matcher
        PSM = propensity_matcher.PropensityScoreMatching(
            ds_month_column=config['ds_month_column'],
            primary_identifier=config['primary_identifier'],
            with_replacement=config['with_replacement'],
            caliper=config['matching_strictness'],
            treatment_flag_col_name='treatment_flag'
        )
        
        # Get months and features
        ls_months = PSM.get_list_month(df=df_selected)
        numerical_features, string_features = PSM.get_feature_type(df=df_selected)
        
        # Run matching for the first month (you can extend this for all months)
        df_matched_pairs = PSM.popensity_matching_monthly(
            df=df_selected,
            month=ls_months[0],
            string_columns=string_features,
            numeric_columns=numerical_features,
            target_columns=['treatment_flag']
        )
        
        # Store matched data
        app.state.data_store['matched_data'] = df_matched_pairs
        
        # Calculate results
        treatment_matched = df_matched_pairs[df_matched_pairs["treatment_flag"] == 1][config['primary_identifier']].nunique()
        control_matched = df_matched_pairs[df_matched_pairs["treatment_flag"] == 0][config['primary_identifier']].nunique()
        
        return JSONResponse(content={
            "success": True,
            "treatment_matched": treatment_matched,
            "control_matched": control_matched,
            "total_matches": len(df_matched_pairs),
            "months_processed": len(ls_months)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running matching: {str(e)}")

@app.get("/download-results")
async def download_results():
    """Download the matched results as CSV"""
    try:
        if 'matched_data' not in app.state.data_store:
            raise HTTPException(status_code=400, detail="No matched data available")
        
        df_matched = app.state.data_store['matched_data']
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        df_matched.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        return JSONResponse(content={
            "success": True,
            "csv_data": csv_content,
            "filename": "matched_pairs.csv"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading results: {str(e)}")

@app.get("/visualizations")
async def get_visualizations():
    """Generate and return visualization plots"""
    try:
        if 'processed_data' not in app.state.data_store or 'matched_data' not in app.state.data_store:
            raise HTTPException(status_code=400, detail="Required data not available")
        
        df_before = app.state.data_store['processed_data']
        df_after = app.state.data_store['matched_data']
        
        # Generate plots using our visualizer
        plots = visualizer.generate_all_plots(
            df_before=df_before,
            df_after=df_after,
            treatment_flag='treatment_flag',
            max_plots=3  # Limit for performance
        )
        
        return JSONResponse(content={
            "success": True,
            "plots": plots
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "mongodb_available": MONGODB_AVAILABLE,
        "version": "1.0.0"
    })

@app.get("/favicon.ico")
async def favicon():
    """Serve a simple favicon to avoid 404 errors"""
    # Return a simple response or you can add an actual favicon file
    return FileResponse("static/favicon.ico") if os.path.exists("static/favicon.ico") else JSONResponse(status_code=204, content=None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 