#!/usr/bin/env python3
"""
Constants for MCA Portfolio Simulator

This module contains constants used throughout the MCA Portfolio Simulator.
"""

# Deal data column names
DEAL_COLUMNS = {
    'ID': 'Funded ID',
    'PRODUCT': 'Product',
    'FUNDING_DATE': 'Initial Funding Date',
    'CREDIT_TIER': 'Credit_Tier',
    'BALANCE': 'Total Original Balance',
    'RTR': 'Total Original RTR',
    'COMMISSION': 'Commission Cost %',
    'FICO': 'Owner FICO',
    'VINTAGE_MONTH': 'Vintage-M',
}

# Payment data column names
PAYMENT_COLUMNS = {
    'ID': 'Funded ID',
    'DATE': 'Transaction Date',
    'FUNDED DATE': 'Funded Date',
    'AMOUNT': 'Transaction Amount',
    'DESCRIPTION': 'Transaction Description',
    'VINTAGE_MONTH': 'Vintage-M',
}

# Transaction types
TRANSACTION_TYPES = {
    'INITIAL_OUTLAY': 'Initial cash outlay',
    'REMITTANCE': 'Merchant remittance',
    'RENEWAL': 'Renewal or discounted payoff adjustment',
    'REFUND': 'Refund',
}

# Portfolio simulation constants
PORTFOLIO = {
    'DEFAULT_SEASONED_DATE': '2024-01-31',
}

# Excel formatting constants
EXCEL = {
    'HEADER_COLOR': 'D9D9D9',
    'HEADER_COLOR_DARK': '4472C4',
    'DATE_FORMAT': 'yyyy-mm-dd',
    'PERCENT_FORMAT': '0.00%',
    'CURRENCY_FORMAT': '"$"#,##0_);("$"#,##0);"-"',
    'COLUMN_WIDTHS': {
        'TRANSACTION_DATE': 15,
        'FUNDED_ID': 15,
        'DESCRIPTION': 30,
        'AMOUNT': 15,
        'PRINCIPAL_REPAID': 15,
    }
}

# Data validation constants
VALIDATION = {
    'COMMISSION_MIN': 0.0,
    'COMMISSION_MAX': 0.20,  # 20% max commission
    'ALLOCATION_MIN': 0.001,  # Minimum 0.1% allocation
    'ALLOCATION_MAX': 1.0,    # Maximum 100% allocation
    'FEE_MIN': 0.0,
    'FEE_MAX': 1.0,
}

# Default file paths
PATHS = {
    'DEFAULT_DATA_FILE': 'data.csv',
    'DEFAULT_PAYMENTS_FILE': 'payments.csv',
    'DEFAULT_OUTPUT_DIR': 'output',
    'DEFAULT_LOG_DIR': 'logs',
}