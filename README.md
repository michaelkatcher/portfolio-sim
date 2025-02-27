# MCA Portfolio Simulator

A Python-based portfolio simulator for analyzing merchant cash advance (MCA) investment strategies.

## Overview

The MCA Portfolio Simulator allows users to create and analyze hypothetical investment portfolios based on historical merchant cash advance data. It enables testing different investment strategies by applying various selection criteria, allocation policies, and fee structures to evaluate their potential performance.

## Features

- **Flexible Deal Selection**: Filter deals by size, vintage period, credit grade, and product type
- **Customizable Allocation Policies**: Define percentage-based allocations with minimum and maximum limits
- **Fee Modeling**: Apply management fees with configurable structures (e.g., after principal repayment)
- **Performance Metrics**: Calculate IRR, MOIC, and other key metrics for the portfolio
- **Excel Output**: Generate detailed Excel workbooks with portfolio analysis results
- **Excel Integration**: Run simulations directly from Excel using the included VBA interface
- **Comprehensive Logging**: Track all operations with detailed logs for audit and debugging

## Installation

### Prerequisites

- Python 3.7+
- Required Python packages (install via pip):
  - pandas
  - numpy
  - openpyxl
  - matplotlib (optional, for potential visualizations)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mca-portfolio-simulator.git
   cd mca-portfolio-simulator
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up data directory:
   ```bash
   mkdir -p raw_data
   ```

5. Place your data files in the `raw_data` directory:
   - `data.csv`: Deal information file
   - `payments.csv`: Payment transactions file

## Usage

### Command Line Interface

Run simulations with various parameters:

```bash
python run_portfolio_analysis.py [options]
```

#### Common Options

- `--min-size VALUE`: Minimum deal size to include
- `--max-size VALUE`: Maximum deal size to include
- `--vintage-start DATE`: Start date for vintage filter (YYYY-MM-DD)
- `--vintage-end DATE`: End date for vintage filter (YYYY-MM-DD)
- `--product-types TYPE1 [TYPE2 ...]`: List of product types to include
- `--allocation-pct VALUE`: Allocation percentage (0-1)
- `--min-allocation VALUE`: Minimum allocation amount
- `--max-allocation VALUE`: Maximum allocation amount
- `--fee-pct VALUE`: Management fee percentage (0-1)
- `--output-dir DIR`: Directory to save output files
- `--output-file FILE`: Output Excel file name
- `--log-file FILE`: Log file name
- `--verbose`: Display detailed information during execution
- `--compare`: Run comparative analysis with multiple scenarios

#### Example

```bash
python run_portfolio_analysis.py --min-size 1000000 --vintage-start 2023-01-01 --vintage-end 2023-12-31 --allocation-pct 0.25 --min-allocation 100000 --max-allocation 500000 --fee-pct 0.02 --output-file portfolio_2023_large.xlsx --verbose
```

### Excel Integration

1. Open the included `portfolio_simulator.xlsm` file
2. Configure simulation parameters using the form controls and input cells
3. Click the "Run Simulation" button
4. Review results in the automatically opened output file

## File Structure

```
mca-portfolio-simulator/
├── raw_data/             # Input data files
│   ├── data.csv          # Deal information
│   └── payments.csv      # Payment transactions
├── output/               # Output Excel files
├── logs/                 # Log files
├── run_portfolio_analysis.py  # Main script
├── data_loader.py        # Data loading module
├── deal_selector.py      # Deal selection module
├── portfolio_simulator.py # Portfolio simulation module
├── excel_exporter.py     # Excel export module
├── common_utils.py       # Common utilities
├── constants.py          # Constants and configuration
└── portfolio_simulator.xlsm  # Excel interface with VBA
```

## Data Format

### data.csv (Deal Information)

Required columns:
- `Funded ID`: Unique identifier for a deal
- `Product`: Internal product type
- `Initial Funding Date`: Date the deal was funded
- `Credit_Tier`: Deal grade
- `Total Original Balance`: Funded amount (principal)
- `Total Original RTR`: Total amount the merchant owes
- `Commission Cost %`: Commission percentage

### payments.csv (Payment Transactions)

Required columns:
- `Funded ID`: Unique identifier for a deal
- `Funded Date`: Date the deal was funded
- `Transaction Date`: Date of the transaction
- `Transaction Amount`: Amount of the transaction
- `Transaction Description`: Type of transaction (e.g., "Initial cash outlay", "Merchant remittance")

## Output Format

The simulator generates an Excel workbook with the following worksheets:

1. **Portfolio Scenario**: Selection criteria, allocation policy, and fee structure
2. **Cashflows**: Individual cashflows for the portfolio
3. **Deals**: List of deals included in the portfolio
4. **Analysis**: Performance metrics including IRR and MOIC

## Development

### Running Tests

```bash
python -m unittest discover tests
```

### Adding New Features

1. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```

2. Make your changes

3. Commit and push:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature-name
   ```

4. Create a pull request on GitHub

## Troubleshooting

### Common Issues

- **File not found errors**: Ensure data files are in the `raw_data` directory
- **Allocation errors**: Check that allocation parameters are reasonable for your deal sizes
- **Performance issues with Excel export**: Consider running directly from terminal rather than through VBA for better performance

### Logs

Check the logs directory for detailed information about each run. Log files are named based on either:
- The timestamp of execution
- A custom name provided via the `--log-file` parameter

## License

[MIT License](LICENSE)

## Contributors

- Your Name - Initial work and ongoing development
- Other Contributors - List others who have contributed

## Acknowledgments

- Thanks to everyone who contributed to this project
- Special thanks to the team who provided the deal and payment data