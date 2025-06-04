// JavaScript for Propensity Matching Tool

let currentStep = 1;
let uploadedData = null;

// Global variable to store the loading modal instance
let loadingModalInstance = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    // File upload handling
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleFileDrop);
    
    // File selection
    fileInput.addEventListener('change', handleFileSelect);

    // Configuration form
    document.getElementById('configForm').addEventListener('submit', handleConfigSubmit);

    // Run matching button
    document.getElementById('runMatchingBtn').addEventListener('click', runMatching);

    // Download button
    document.getElementById('downloadBtn').addEventListener('click', downloadResults);

    // Visualizations button
    document.getElementById('showVisualizationsBtn').addEventListener('click', showVisualizations);
}

// File handling functions
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

async function processFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showAlert('Please select a CSV file.', 'danger');
        return;
    }

    showLoading('Uploading and processing file...');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload-data', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        hideLoading();

        if (result.success) {
            uploadedData = result.summary;
            displayDataSummary(result.summary);
            populateColumnSelectors(result.summary);
            moveToStep(2);
        } else {
            showAlert('Error uploading file: ' + result.detail, 'danger');
        }
    } catch (error) {
        hideLoading();
        showAlert('Error uploading file: ' + error.message, 'danger');
    } finally {
        // Ensure loading modal is hidden regardless of success/failure
        setTimeout(hideLoading, 100);
    }
}

function displayDataSummary(summary) {
    const fileInfo = document.getElementById('fileInfo');
    const dataSummary = document.getElementById('dataSummary');
    
    dataSummary.innerHTML = `
        <div class="row mt-3">
            <div class="col-md-6">
                <strong>Shape:</strong> ${summary.shape[0]} rows × ${summary.shape[1]} columns<br>
                <strong>Numeric columns:</strong> ${summary.numeric_columns.length}<br>
                <strong>String columns:</strong> ${summary.string_columns.length}
            </div>
            <div class="col-md-6">
                <strong>Missing values:</strong> ${Object.values(summary.missing_values).reduce((a, b) => a + b, 0)} total<br>
                <strong>ID columns found:</strong> ${summary.id_columns.length}
            </div>
        </div>
        <div class="mt-3">
            <strong>Columns:</strong> ${summary.columns.join(', ')}
        </div>
    `;
    
    fileInfo.style.display = 'block';
}

function populateColumnSelectors(summary) {
    const primaryIdentifier = document.getElementById('primaryIdentifier');
    const treatmentFlag = document.getElementById('treatmentFlag');
    const dsMonth = document.getElementById('dsMonth');
    const matchingColumns = document.getElementById('matchingColumns');

    // Clear existing options
    primaryIdentifier.innerHTML = '<option value="">Select column...</option>';
    treatmentFlag.innerHTML = '<option value="">Select column...</option>';
    dsMonth.innerHTML = '<option value="">Select column...</option>';
    matchingColumns.innerHTML = '';

    // Populate dropdowns
    summary.columns.forEach(column => {
        // Primary identifier - prefer ID columns
        const primaryOption = new Option(column, column);
        if (summary.id_columns.includes(column)) {
            primaryOption.selected = true;
        }
        primaryIdentifier.appendChild(primaryOption);

        // Treatment flag - prefer boolean-looking columns
        const treatmentOption = new Option(column, column);
        if (column.toLowerCase().includes('treatment') || column.toLowerCase().includes('flag')) {
            treatmentOption.selected = true;
        }
        treatmentFlag.appendChild(treatmentOption);

        // Date/month column - prefer date-looking columns
        const dateOption = new Option(column, column);
        if (column.toLowerCase().includes('date') || column.toLowerCase().includes('month') || column.toLowerCase().includes('ds')) {
            dateOption.selected = true;
        }
        dsMonth.appendChild(dateOption);
    });

    // Populate matching columns checkboxes
    const excludeColumns = [...summary.id_columns];
    summary.columns.forEach(column => {
        if (!excludeColumns.includes(column)) {
            const checkboxDiv = document.createElement('div');
            checkboxDiv.className = 'form-check';
            checkboxDiv.innerHTML = `
                <input class="form-check-input" type="checkbox" value="${column}" id="match_${column}">
                <label class="form-check-label" for="match_${column}">
                    ${column} (${summary.data_types[column]})
                </label>
            `;
            matchingColumns.appendChild(checkboxDiv);
        }
    });
}

async function handleConfigSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData();
    formData.append('primary_identifier', document.getElementById('primaryIdentifier').value);
    formData.append('treatment_flag', document.getElementById('treatmentFlag').value);
    formData.append('ds_month_column', document.getElementById('dsMonth').value);
    formData.append('handle_missing', document.getElementById('handleMissing').value);
    formData.append('standardize_numeric', document.getElementById('standardizeNumeric').checked);
    formData.append('matching_strictness', document.getElementById('matchingStrictness').value);
    formData.append('with_replacement', document.getElementById('withReplacement').checked);

    // Get selected matching columns
    const selectedColumns = [];
    document.querySelectorAll('#matchingColumns input:checked').forEach(checkbox => {
        selectedColumns.push(checkbox.value);
    });
    
    if (selectedColumns.length === 0) {
        showAlert('Please select at least one column for matching.', 'warning');
        return;
    }
    
    formData.append('columns_for_matching', selectedColumns.join(','));

    showLoading('Configuring matching parameters...');

    try {
        const response = await fetch('/configure-matching', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        hideLoading();

        if (result.success) {
            displayConfigResults(result);
            moveToStep(3);
        } else {
            showAlert('Configuration error: ' + result.detail, 'danger');
        }
    } catch (error) {
        hideLoading();
        showAlert('Configuration error: ' + error.message, 'danger');
    } finally {
        // Ensure loading modal is hidden regardless of success/failure
        setTimeout(hideLoading, 100);
    }
}

function displayConfigResults(result) {
    const configResults = document.getElementById('configResults');
    configResults.innerHTML = `
        <div class="alert alert-success">
            <h6><i class="fas fa-check-circle me-2"></i>Configuration Successful</h6>
            <div class="row">
                <div class="col-md-6">
                    <strong>Treatment Group Size:</strong> ${result.treatment_size}<br>
                    <strong>Control Group Size:</strong> ${result.control_size}
                </div>
                <div class="col-md-6">
                    <strong>Processed Data Shape:</strong> ${result.processed_shape[0]} × ${result.processed_shape[1]}<br>
                    <strong>Validation:</strong> <span class="text-success">${result.validation}</span>
                </div>
            </div>
        </div>
    `;
    configResults.style.display = 'block';
}

async function runMatching() {
    document.getElementById('matchingControls').style.display = 'none';
    document.getElementById('matchingProgress').style.display = 'block';

    try {
        const response = await fetch('/run-matching', {
            method: 'POST'
        });

        const result = await response.json();

        document.getElementById('matchingProgress').style.display = 'none';

        if (result.success) {
            displayMatchingResults(result);
            document.getElementById('matchingComplete').style.display = 'block';
            moveToStep(4);
        } else {
            showAlert('Matching error: ' + result.detail, 'danger');
            document.getElementById('matchingControls').style.display = 'block';
        }
    } catch (error) {
        document.getElementById('matchingProgress').style.display = 'none';
        document.getElementById('matchingControls').style.display = 'block';
        showAlert('Matching error: ' + error.message, 'danger');
    }
}

function displayMatchingResults(result) {
    const matchingResults = document.getElementById('matchingResults');
    matchingResults.innerHTML = `
        <div class="row">
            <div class="col-md-4">
                <strong>Treatment Matched:</strong> ${result.treatment_matched}
            </div>
            <div class="col-md-4">
                <strong>Control Matched:</strong> ${result.control_matched}
            </div>
            <div class="col-md-4">
                <strong>Total Matches:</strong> ${result.total_matches}
            </div>
        </div>
        <div class="mt-2">
            <strong>Months Processed:</strong> ${result.months_processed}
        </div>
    `;
}

async function downloadResults() {
    showLoading('Preparing download...');

    try {
        const response = await fetch('/download-results');
        const result = await response.json();
        hideLoading();

        if (result.success) {
            // Create and trigger download
            const blob = new Blob([result.csv_data], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = result.filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showAlert('Results downloaded successfully!', 'success');
        } else {
            showAlert('Download error: ' + result.detail, 'danger');
        }
    } catch (error) {
        hideLoading();
        showAlert('Download error: ' + error.message, 'danger');
    } finally {
        // Ensure loading modal is hidden regardless of success/failure
        setTimeout(hideLoading, 100);
    }
}

async function showVisualizations() {
    showLoading('Generating visualizations...');

    try {
        const response = await fetch('/visualizations');
        const result = await response.json();
        hideLoading();

        if (result.success) {
            displayPlots(result.plots);
            document.getElementById('visualizations').style.display = 'block';
        } else {
            showAlert('Visualization error: ' + result.detail, 'danger');
        }
    } catch (error) {
        hideLoading();
        showAlert('Visualization error: ' + error.message, 'danger');
    } finally {
        // Ensure loading modal is hidden regardless of success/failure
        setTimeout(hideLoading, 100);
    }
}

function displayPlots(plots) {
    const plotsContainer = document.getElementById('plotsContainer');
    plotsContainer.innerHTML = '';

    plots.forEach(plot => {
        const plotDiv = document.createElement('div');
        plotDiv.className = 'plot-container mb-4';
        plotDiv.innerHTML = `
            <h5>${plot.column}</h5>
            <img src="data:image/png;base64,${plot.plot}" alt="Distribution plot for ${plot.column}" class="img-fluid">
        `;
        plotsContainer.appendChild(plotDiv);
    });
}

// Step navigation functions
function moveToStep(stepNumber) {
    // Hide all step content
    document.querySelectorAll('.step-content').forEach(content => {
        content.style.display = 'none';
    });

    // Show current step content
    document.getElementById(`content-step${stepNumber}`).style.display = 'block';

    // Update step indicators
    updateStepIndicators(stepNumber);
    
    currentStep = stepNumber;
}

function updateStepIndicators(currentStepNumber) {
    document.querySelectorAll('.step').forEach((step, index) => {
        const stepNumber = index + 1;
        step.classList.remove('active', 'completed');
        
        if (stepNumber === currentStepNumber) {
            step.classList.add('active');
        } else if (stepNumber < currentStepNumber) {
            step.classList.add('completed');
        }
    });
}

// Utility functions
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the current step content
    const currentStepContent = document.getElementById(`content-step${currentStep}`);
    currentStepContent.insertBefore(alertDiv, currentStepContent.firstChild);
    
    // Auto-dismiss success and info alerts after 5 seconds
    if (type === 'success' || type === 'info') {
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
}

function showLoading(message = 'Processing...') {
    document.getElementById('loadingMessage').textContent = message;
    
    // Get or create the modal instance
    const modalElement = document.getElementById('loadingModal');
    loadingModalInstance = bootstrap.Modal.getInstance(modalElement) || new bootstrap.Modal(modalElement, {
        backdrop: 'static',  // Prevent closing by clicking backdrop
        keyboard: false      // Prevent closing with Esc key during processing
    });
    
    loadingModalInstance.show();
}

function hideLoading() {
    // Try multiple methods to ensure the modal is hidden
    if (loadingModalInstance) {
        loadingModalInstance.hide();
        loadingModalInstance = null;
    } else {
        // Fallback: try to get existing instance
        const modalElement = document.getElementById('loadingModal');
        const existingInstance = bootstrap.Modal.getInstance(modalElement);
        if (existingInstance) {
            existingInstance.hide();
        } else {
            // Force hide by manipulating the DOM directly
            modalElement.classList.remove('show');
            modalElement.style.display = 'none';
            modalElement.setAttribute('aria-hidden', 'true');
            modalElement.removeAttribute('aria-modal');
            modalElement.removeAttribute('role');
            
            // Remove backdrop if it exists
            const backdrop = document.querySelector('.modal-backdrop');
            if (backdrop) {
                backdrop.remove();
            }
            
            // Remove modal-open class from body
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
        }
    }
    
    // Additional cleanup to ensure UI is responsive
    setTimeout(() => {
        document.body.classList.remove('modal-open');
        document.body.style.overflow = '';
        document.body.style.paddingRight = '';
        
        // Remove any remaining backdrops
        document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
            backdrop.remove();
        });
    }, 100);
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to proceed to next step
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const nextButton = document.querySelector(`#content-step${currentStep} button[type="submit"], #content-step${currentStep} .btn-success`);
        if (nextButton && !nextButton.disabled) {
            nextButton.click();
        }
    }
});
