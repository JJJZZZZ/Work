# Propensity Matching Tool - Web Application

A modern, production-ready web application for propensity score matching analysis. This tool provides an intuitive interface for uploading data, configuring matching parameters, and visualizing results.

## Features

- **Easy Data Upload**: Drag-and-drop CSV file upload with automatic data validation
- **Interactive Configuration**: Smart column detection and parameter configuration
- **Advanced Matching**: Propensity score matching with customizable parameters
- **Visual Analysis**: Before/after matching comparison plots
- **Export Results**: Download matched pairs as CSV files
- **Responsive Design**: Works on desktop and mobile devices

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd "Matching Tool/Hosting Web"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

1. **Run the FastAPI server**:
   ```bash
   python app.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:8000
   ```

### Using the Tool

The application follows a 4-step workflow:

#### Step 1: Upload Data
- Drag and drop your CSV file or click to browse
- The tool will automatically analyze your data and display summary statistics
- Supported format: CSV files with headers

#### Step 2: Configure Matching
- **Primary Identifier**: Select the column that uniquely identifies each observation
- **Treatment Flag**: Select the binary column indicating treatment (1) vs control (0)
- **Date/Month Column**: Select the time period column
- **Matching Columns**: Choose which variables to use for propensity matching
- **Processing Options**:
  - Missing value handling (drop, fill with mean/median)
  - Numeric variable standardization
  - Matching strictness (caliper)
  - With/without replacement

#### Step 3: Run Matching
- Click "Run Matching" to execute the propensity score matching algorithm
- The process may take a few moments depending on data size
- View matching results including group sizes and match counts

#### Step 4: Results and Analysis
- **Download Results**: Export matched pairs as CSV
- **Visualizations**: View before/after matching distribution plots
- **Balance Assessment**: Analyze matching quality with standardized mean differences

## Data Requirements

Your CSV file should contain:

1. **Primary Identifier**: A unique identifier for each observation (e.g., `user_id`, `org_id`)
2. **Treatment Flag**: A binary variable (0/1 or True/False) indicating treatment assignment
3. **Time Variable**: A date or month column for temporal matching
4. **Matching Variables**: Continuous or categorical variables to use for propensity matching

### Example Data Structure

```csv
org_id,region,channel_group,mrr,treatment_flag,ds_month
123,US,Online,1000,1,2024-01
456,EU,Sales,1500,0,2024-01
789,US,Online,800,1,2024-01
...
```

## API Endpoints

The application provides several REST API endpoints:

- `GET /`: Main application interface
- `POST /upload-data`: Upload and validate CSV data
- `POST /configure-matching`: Configure matching parameters
- `POST /run-matching`: Execute propensity matching
- `GET /download-results`: Download matched results
- `GET /visualizations`: Generate comparison plots

## Configuration Options

### Matching Parameters

- **Caliper (Matching Strictness)**: Controls how close propensity scores must be for matching (default: 0.1)
- **With Replacement**: Whether control units can be matched to multiple treatment units
- **Standardization**: Whether to standardize numeric variables before matching

### Data Processing

- **Missing Values**: Options to drop rows, fill with mean, or fill with median
- **Outlier Handling**: Automatic winsorization using IQR method
- **Control Group Sampling**: Option to sample control group for faster processing

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**: Change the port in `app.py`:
   ```python
   uvicorn.run(app, host="0.0.0.0", port=8001)  # Use different port
   ```

3. **Large File Processing**: For very large files, consider:
   - Sampling your data before upload
   - Using the control group sampling feature
   - Increasing server timeout settings

4. **MongoDB Dependencies**: If you encounter MongoDB package errors:
   - These packages are optional
   - Comment out the MongoDB imports in `app.py`
   - The core functionality will still work

### Performance Tips

- **Data Size**: The tool works best with datasets under 100,000 rows
- **Memory Usage**: Close the browser tab when not in use to free memory
- **Processing Time**: Matching time depends on data size and number of variables

## Development

### Project Structure

```
Matching Tool/Hosting Web/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── static/               # Frontend assets
│   ├── index.html        # Main HTML interface
│   ├── style.css         # CSS styling
│   └── script.js         # JavaScript functionality
└── core/                 # Backend modules
    ├── data_processor.py # Data processing logic
    └── visualizations.py # Plot generation
```

### Adding Features

1. **New Endpoints**: Add to `app.py`
2. **Frontend Changes**: Modify files in `static/`
3. **Data Processing**: Extend `core/data_processor.py`
4. **Visualizations**: Add plots to `core/visualizations.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is intended for internal use and research purposes.

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the console output for error messages
3. Contact the development team

---

**Built with**: FastAPI, Pandas, Scikit-learn, Matplotlib, Bootstrap
