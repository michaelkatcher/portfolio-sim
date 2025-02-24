#!/usr/bin/env python3
"""
Excel Exporter for MCA Portfolio Simulator

This module handles the exporting of portfolio simulation results to Excel,
creating a formatted workbook with multiple sheets as specified in the project requirements.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment, numbers
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.marker import Marker


def export_portfolio_to_excel(portfolio, filename, verbose=False):
    """
    Export a portfolio simulation to Excel with formatted sheets.
    
    Args:
        portfolio: A simulated Portfolio instance
        filename: The output Excel file path
        verbose: Whether to print progress information
        
    Returns:
        Path to the created Excel file
    """
    if verbose:
        print(f"Exporting portfolio '{portfolio.name}' to Excel: {filename}")
    
    # Create a workbook
    wb = Workbook()
    
    # Remove the default sheet
    default_sheet = wb.active
    wb.remove(default_sheet)
    
    # Create each sheet
    _create_portfolio_scenario_sheet(wb, portfolio)
    _create_cashflows_sheet(wb, portfolio)
    _create_deals_sheet(wb, portfolio)
    _create_analysis_sheet(wb, portfolio)
    
    # Save the workbook
    wb.save(filename)
    
    if verbose:
        print(f"Export completed: {filename}")
    
    return filename


# Update to the _create_portfolio_scenario_sheet function to include commission statistics

def _create_portfolio_scenario_sheet(wb, portfolio):
    """
    Create the Portfolio Scenario worksheet.
    
    Args:
        wb: Workbook object
        portfolio: Portfolio instance
    """
    ws = wb.create_sheet("Portfolio Scenario", 0)
    
    # Define styles
    header_font = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="D9D9D9")
    
    # Set column widths
    ws.column_dimensions['A'].width = 30
    ws.column_dimensions['B'].width = 50
    
    # Add title
    ws['A1'] = "Portfolio Scenario"
    ws['A1'].font = Font(bold=True, size=14)
    
    # Add selection criteria section
    row = 3
    ws['A' + str(row)] = "Selection Criteria"
    ws['A' + str(row)].font = header_font
    ws.cell(row=row, column=1).fill = header_fill
    ws.cell(row=row, column=2).fill = header_fill
    
    row += 1
    ws['A' + str(row)] = "Product Types"
    if 'product_types' in portfolio.selection_criteria and portfolio.selection_criteria['product_types']:
        ws['B' + str(row)] = ", ".join(portfolio.selection_criteria['product_types'])
    else:
        ws['B' + str(row)] = "All"
        
    row += 1
    ws['A' + str(row)] = "Credit Grades"
    if 'credit_tiers' in portfolio.selection_criteria and portfolio.selection_criteria['credit_tiers']:
        ws['B' + str(row)] = ", ".join(portfolio.selection_criteria['credit_tiers'])
    else:
        ws['B' + str(row)] = "All"
        
    row += 1
    ws['A' + str(row)] = "Deal Size Range"
    if 'deal_size_range' in portfolio.selection_criteria and portfolio.selection_criteria['deal_size_range']:
        min_size, max_size = portfolio.selection_criteria['deal_size_range']
        ws['B' + str(row)] = f"${min_size:,.2f} to ${max_size:,.2f}"
    else:
        ws['B' + str(row)] = "All"
        
    row += 1
    ws['A' + str(row)] = "Vintages"
    if 'vintage_range' in portfolio.selection_criteria and portfolio.selection_criteria['vintage_range']:
        start_date, end_date = portfolio.selection_criteria['vintage_range']
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        ws['B' + str(row)] = f"{start_date} to {end_date}"
    else:
        ws['B' + str(row)] = "All"
        
    # Add allocation policy section
    row += 2
    ws['A' + str(row)] = "Allocation Policy"
    ws['A' + str(row)].font = header_font
    ws.cell(row=row, column=1).fill = header_fill
    ws.cell(row=row, column=2).fill = header_fill
    
    row += 1
    ws['A' + str(row)] = "Allocation Formula"
    # Format for the MAX(min, MIN(max, pct*DealSize)) formula
    min_amt = portfolio.allocation_policy.min_amount or 0
    max_amt = portfolio.allocation_policy.max_amount or "∞"
    pct = portfolio.allocation_policy.percentage * 100
    
    if min_amt > 0 and max_amt != "∞":
        formula_str = f"MAX({min_amt:,.0f}, MIN({max_amt:,.0f}, {pct:.1f}%*DealSize))"
    elif min_amt > 0:
        formula_str = f"MAX({min_amt:,.0f}, {pct:.1f}%*DealSize)"
    elif max_amt != "∞":
        formula_str = f"MIN({max_amt:,.0f}, {pct:.1f}%*DealSize)"
    else:
        formula_str = f"{pct:.1f}%*DealSize"
        
    ws['B' + str(row)] = formula_str
    
    # Add fee structure section
    row += 2
    ws['A' + str(row)] = "Fee Structure"
    ws['A' + str(row)].font = header_font
    ws.cell(row=row, column=1).fill = header_fill
    ws.cell(row=row, column=2).fill = header_fill
    
    row += 1
    ws['A' + str(row)] = "Management Fee"
    fee_pct = portfolio.fee_schedule.percentage * 100
    if portfolio.fee_schedule.apply_after_principal:
        ws['B' + str(row)] = f"{fee_pct:.1f}% of incoming cashflows after principal repayment"
    else:
        ws['B' + str(row)] = f"{fee_pct:.1f}% of all incoming cashflows"
    
    # Add commission section if data is available
    if portfolio.deal_allocations is not None and 'Commission Cost %' in portfolio.deal_allocations.columns:
        row += 2
        ws['A' + str(row)] = "Commission Information"
        ws['A' + str(row)].font = header_font
        ws.cell(row=row, column=1).fill = header_fill
        ws.cell(row=row, column=2).fill = header_fill
        
        row += 1
        ws['A' + str(row)] = "Average Commission Rate"
        avg_commission = portfolio.deal_allocations['Commission Cost %'].mean() * 100
        ws['B' + str(row)] = f"{avg_commission:.2f}%"
        
        row += 1
        ws['A' + str(row)] = "Weighted Avg Commission Rate"
        # Calculate weighted average by deal size
        weighted_avg = (portfolio.deal_allocations['Commission Cost %'] * 
                         portfolio.deal_allocations['Total Original Balance']).sum() / portfolio.deal_allocations['Total Original Balance'].sum() * 100
        ws['B' + str(row)] = f"{weighted_avg:.2f}%"
        
        row += 1
        ws['A' + str(row)] = "Min Commission Rate"
        min_commission = portfolio.deal_allocations['Commission Cost %'].min() * 100
        ws['B' + str(row)] = f"{min_commission:.2f}%"
        
        row += 1
        ws['A' + str(row)] = "Max Commission Rate"
        max_commission = portfolio.deal_allocations['Commission Cost %'].max() * 100
        ws['B' + str(row)] = f"{max_commission:.2f}%"
        
        row += 1
        ws['A' + str(row)] = "Deals with 0% Commission"
        zero_commission = (portfolio.deal_allocations['Commission Cost %'] == 0).sum()
        ws['B' + str(row)] = f"{zero_commission} ({(zero_commission / len(portfolio.deal_allocations)) * 100:.1f}%)"
    
    # Add portfolio summary section if metrics are available
    if portfolio.metrics:
        row += 2
        ws['A' + str(row)] = "Portfolio Summary"
        ws['A' + str(row)].font = header_font
        ws.cell(row=row, column=1).fill = header_fill
        ws.cell(row=row, column=2).fill = header_fill
        
        row += 1
        ws['A' + str(row)] = "Total Deals"
        ws['B' + str(row)] = f"{portfolio.metrics['all']['deal_count']:,}"
        
        row += 1
        ws['A' + str(row)] = "Total Allocated Amount"
        if portfolio.deal_allocations is not None:
            total_allocation = portfolio.deal_allocations['Allocation Amount'].sum()
            ws['B' + str(row)] = f"${total_allocation:,.2f}"
        
        row += 1
        ws['A' + str(row)] = "Total Investment"
        ws['B' + str(row)] = f"${portfolio.metrics['all']['total_invested']:,.2f}"
        
        row += 1
        ws['A' + str(row)] = "Total Return"
        ws['B' + str(row)] = f"${portfolio.metrics['all']['total_returned']:,.2f}"
        
        row += 1
        ws['A' + str(row)] = "MOIC"
        ws['B' + str(row)] = f"{portfolio.metrics['all']['moic']:.2f}x"
        
        row += 1
        ws['A' + str(row)] = "IRR"
        if not np.isnan(portfolio.metrics['all'].get('irr', float('nan'))):
            ws['B' + str(row)] = f"{portfolio.metrics['all']['irr']*100:.2f}%"
        else:
            ws['B' + str(row)] = "N/A"
    
    return ws


def _create_cashflows_sheet(wb, portfolio):
    """
    Create the Cashflows worksheet with all individual cashflows.
    
    Args:
        wb: Workbook object
        portfolio: Portfolio instance
    """
    ws = wb.create_sheet("Cashflows", 1)
    
    if portfolio.cashflows is None:
        ws['A1'] = "No cashflow data available"
        return ws
    
    # Define styles
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="4472C4")
    money_format = numbers.FORMAT_CURRENCY_USD_SIMPLE
    
    # Set column widths
    column_widths = {
        'A': 15,  # Transaction Date
        'B': 15,  # Funded ID
        'C': 30,  # Transaction Description
        'D': 15,  # Portfolio Amount
        'E': 15,  # Fee Amount
        'F': 15,  # Net Amount
        'G': 10,  # Principal Repaid
    }
    
    for col, width in column_widths.items():
        ws.column_dimensions[col].width = width
    
    # Define columns to include
    columns = ['Transaction Date', 'Funded ID', 'Transaction Description', 
               'Portfolio Amount', 'Fee Amount', 'Net Amount', 'Principal Repaid']
    
    # Add headers
    for col_idx, col_name in enumerate(columns, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center')
    
    # Prepare cashflow data
    cf_data = portfolio.cashflows.copy()
    
    # Sort by Transaction Date
    cf_data = cf_data.sort_values('Transaction Date')
    
    # Convert dates to strings for Excel
    cf_data['Transaction Date'] = cf_data['Transaction Date'].dt.strftime('%Y-%m-%d')
    
    # Add data rows
    for row_idx, row_data in enumerate(dataframe_to_rows(cf_data[columns], index=False, header=False), 2):
        for col_idx, value in enumerate(row_data, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            
            # Format currency columns
            if col_idx in [4, 5, 6]:  # Amount columns
                cell.number_format = money_format
            
            # Format boolean column
            if col_idx == 7:  # Principal Repaid
                cell.value = "Yes" if value else "No"
                cell.alignment = Alignment(horizontal='center')
    
    # Add totals row at the bottom
    row_idx = len(cf_data) + 2
    ws.cell(row=row_idx, column=1, value="TOTAL")
    ws.cell(row=row_idx, column=1).font = Font(bold=True)
    
    # Sum amount columns
    for col_idx, col_name in [(4, 'Portfolio Amount'), (5, 'Fee Amount'), (6, 'Net Amount')]:
        total = cf_data[col_name].sum()
        cell = ws.cell(row=row_idx, column=col_idx, value=total)
        cell.font = Font(bold=True)
        cell.number_format = money_format
    
    return ws


def _create_deals_sheet(wb, portfolio):
    """
    Create the Deals worksheet with all individual deals in the portfolio.
    
    Args:
        wb: Workbook object
        portfolio: Portfolio instance
    """
    ws = wb.create_sheet("Deals", 2)
    
    if portfolio.deal_allocations is None:
        ws['A1'] = "No deal data available"
        return ws
    
    # Define styles
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="4472C4")
    money_format = numbers.FORMAT_CURRENCY_USD_SIMPLE
    date_format = 'yyyy-mm-dd'
    percent_format = '0.00%'
    
    # Set column widths
    column_widths = {
        'A': 15,  # Funded ID
        'B': 15,  # Product
        'C': 15,  # Initial Funding Date
        'D': 12,  # Credit Tier
        'E': 15,  # Total Original Balance
        'F': 15,  # Total Original RTR
        'G': 15,  # Commission Cost %
        'H': 15,  # Allocation Amount
        'I': 15,  # Allocation Percentage
        'J': 15,  # Allocated RTR
    }
    
    for col, width in column_widths.items():
        ws.column_dimensions[col].width = width
    
    # Define columns to include - add Commission Cost % to the list
    columns = [
        'Funded ID', 'Product', 'Initial Funding Date', 'Credit_Tier', 
        'Total Original Balance', 'Total Original RTR', 'Commission Cost %',
        'Allocation Amount', 'Allocation Percentage', 'Allocated RTR'
    ]
    
    # Check which columns are actually available
    available_columns = [col for col in columns if col in portfolio.deal_allocations.columns]
    
    # Add headers
    for col_idx, col_name in enumerate(available_columns, 1):
        # Use more readable header names
        display_name = col_name.replace('_', ' ')
        cell = ws.cell(row=1, column=col_idx, value=display_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center')
    
    # Prepare deal data
    deal_data = portfolio.deal_allocations[available_columns].copy()
    
    # Sort by Initial Funding Date if available
    if 'Initial Funding Date' in deal_data.columns:
        deal_data = deal_data.sort_values('Initial Funding Date')
        # Convert dates to strings for Excel
        deal_data['Initial Funding Date'] = deal_data['Initial Funding Date'].dt.strftime('%Y-%m-%d')
    
    # Add data rows
    for row_idx, row_data in enumerate(dataframe_to_rows(deal_data, index=False, header=False), 2):
        for col_idx, value in enumerate(row_data, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            
            # Format based on column type
            col_name = available_columns[col_idx-1]
            
            # Format currency columns
            if col_name in ['Total Original Balance', 'Total Original RTR', 'Allocation Amount', 'Allocated RTR']:
                cell.number_format = money_format
            
            # Format percentage columns
            if col_name in ['Allocation Percentage', 'Commission Cost %']:
                cell.number_format = percent_format
    
    # Add totals row at the bottom
    row_idx = len(deal_data) + 2
    ws.cell(row=row_idx, column=1, value="TOTAL")
    ws.cell(row=row_idx, column=1).font = Font(bold=True)
    
    # Sum amount columns
    for col_idx, col_name in enumerate(available_columns, 1):
        if col_name in ['Total Original Balance', 'Total Original RTR', 'Allocation Amount', 'Allocated RTR']:
            total = deal_data[col_name].sum()
            cell = ws.cell(row=row_idx, column=col_idx, value=total)
            cell.font = Font(bold=True)
            cell.number_format = money_format
        elif col_name == 'Commission Cost %':
            # For Commission Cost %, calculate the weighted average by deal size
            if 'Total Original Balance' in deal_data.columns:
                weighted_avg = (deal_data['Commission Cost %'] * deal_data['Total Original Balance']).sum() / deal_data['Total Original Balance'].sum()
                cell = ws.cell(row=row_idx, column=col_idx, value=weighted_avg)
                cell.font = Font(bold=True)
                cell.number_format = percent_format
    
    return ws


def _create_analysis_sheet(wb, portfolio):
    """
    Create the Analysis worksheet with IRR and MOIC calculations.
    
    Args:
        wb: Workbook object
        portfolio: Portfolio instance
    """
    ws = wb.create_sheet("Analysis", 3)
    
    if portfolio.metrics is None:
        ws['A1'] = "No analysis data available"
        return ws
    
    # Define styles
    header_font = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="D9D9D9")
    subheader_fill = PatternFill("solid", fgColor="E6E6E6")
    money_format = numbers.FORMAT_CURRENCY_USD_SIMPLE
    
    # Set column widths
    for col in ['A', 'B', 'C']:
        ws.column_dimensions[col].width = 20
    
    # Add title
    ws['A1'] = "Portfolio Performance Analysis"
    ws['A1'].font = Font(bold=True, size=14)
    
    # Overall performance section
    row = 3
    ws['A' + str(row)] = "Overall Performance"
    ws['A' + str(row)].font = header_font
    ws.merge_cells(f'A{row}:C{row}')
    ws.cell(row=row, column=1).fill = header_fill
    
    row += 1
    ws['A' + str(row)] = "Metric"
    ws['B' + str(row)] = "Value"
    for col in range(1, 3):
        ws.cell(row=row, column=col).font = header_font
        ws.cell(row=row, column=col).fill = subheader_fill
    
    # Add overall metrics
    metrics = [
        ('Deal Count', portfolio.metrics['all']['deal_count']),
        ('Total Invested', portfolio.metrics['all']['total_invested']),
        ('Total Returned', portfolio.metrics['all']['total_returned']),
        ('Net Cashflow', portfolio.metrics['all']['net_cashflow']),
        ('MOIC', portfolio.metrics['all']['moic']),
        ('IRR', portfolio.metrics['all'].get('irr', float('nan'))),
    ]
    
    for metric_name, metric_value in metrics:
        row += 1
        ws['A' + str(row)] = metric_name
        
        if metric_name in ['Total Invested', 'Total Returned', 'Net Cashflow']:
            ws.cell(row=row, column=2, value=metric_value)
            ws.cell(row=row, column=2).number_format = money_format
        elif metric_name == 'MOIC':
            ws.cell(row=row, column=2, value=metric_value)
            ws.cell(row=row, column=2).number_format = '0.00x'
        elif metric_name == 'IRR':
            if not np.isnan(metric_value):
                ws.cell(row=row, column=2, value=metric_value)
                ws.cell(row=row, column=2).number_format = '0.00%'
            else:
                ws.cell(row=row, column=2, value="N/A")
        else:
            ws.cell(row=row, column=2, value=metric_value)
    
    # Vintage Performance section - calculate IRR and MOIC by vintage month
    if portfolio.cashflows is not None and portfolio.deal_allocations is not None:
        # Add a title for the vintage analysis section
        row += 2
        ws['A' + str(row)] = "Performance by Monthly Vintage"
        ws['A' + str(row)].font = header_font
        ws.merge_cells(f'A{row}:C{row}')
        ws.cell(row=row, column=1).fill = header_fill
        
        row += 1
        headers = ['Vintage', 'IRR', 'MOIC']
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col_idx, value=header)
            cell.font = header_font
            cell.fill = subheader_fill
        
        # Calculate performance by vintage month
        try:
            # Check if we have 'Vintage-M' in the cashflows data
            if 'Vintage-M' in portfolio.cashflows.columns:
                # Group cashflows by vintage and calculate metrics
                vintage_data = []
                
                # Get unique vintages
                unique_vintages = portfolio.cashflows['Vintage-M'].unique()
                
                for vintage in sorted(unique_vintages):
                    # Get cashflows for this vintage
                    vintage_cf = portfolio.cashflows[portfolio.cashflows['Vintage-M'] == vintage]
                    
                    # Group by date and sum amounts
                    cf_by_date = vintage_cf.groupby('Transaction Date')['Net Amount'].sum().reset_index()
                    cf_by_date = cf_by_date.sort_values('Transaction Date')
                    
                    # Calculate metrics
                    total_in = vintage_cf[vintage_cf['Net Amount'] > 0]['Net Amount'].sum()
                    total_out = abs(vintage_cf[vintage_cf['Net Amount'] < 0]['Net Amount'].sum())
                    
                    if total_out > 0:
                        moic = total_in / total_out
                    else:
                        moic = float('nan')
                    
                    # Calculate IRR if possible
                    cf_amounts = cf_by_date['Net Amount'].tolist()
                    irr = float('nan')
                    
                    if len(cf_amounts) > 1 and not all(amt >= 0 for amt in cf_amounts):
                        try:
                            # Use numpy's IRR function
                            monthly_irr = np.irr(cf_amounts)
                            # Annualize the IRR
                            irr = (1 + monthly_irr) ** 12 - 1
                        except:
                            pass
                    
                    vintage_data.append({
                        'Vintage': vintage,
                        'IRR': irr,
                        'MOIC': moic,
                        'Invested': total_out,
                        'Returned': total_in
                    })
                
                # Sort by vintage
                vintage_data = sorted(vintage_data, key=lambda x: x['Vintage'])
                
                # Add vintage data rows
                for vintage_metrics in vintage_data:
                    row += 1
                    ws.cell(row=row, column=1, value=vintage_metrics['Vintage'])
                    
                    # IRR
                    if not np.isnan(vintage_metrics['IRR']):
                        ws.cell(row=row, column=2, value=vintage_metrics['IRR'])
                        ws.cell(row=row, column=2).number_format = '0.00%'
                    else:
                        ws.cell(row=row, column=2, value="N/A")
                    
                    # MOIC
                    if not np.isnan(vintage_metrics['MOIC']):
                        ws.cell(row=row, column=3, value=vintage_metrics['MOIC'])
                        ws.cell(row=row, column=3).number_format = '0.00x'
                    else:
                        ws.cell(row=row, column=3, value="N/A")
                
                # Add vintage performance charts
                if len(vintage_data) > 1:
                    # Add IRR chart
                    _add_vintage_chart(
                        ws, 
                        vintage_data, 
                        "IRR by Vintage", 
                        "IRR", 
                        'E5', 
                        percentage=True
                    )
                    
                    # Add MOIC chart
                    _add_vintage_chart(
                        ws, 
                        vintage_data, 
                        "MOIC by Vintage", 
                        "MOIC", 
                        'E20', 
                        percentage=False
                    )
            
        except Exception as e:
            # Handle any errors in vintage performance calculation
            row += 1
            ws.cell(row=row, column=1, value=f"Error calculating vintage performance: {str(e)}")
    
    return ws


def _add_vintage_chart(ws, vintage_data, title, metric_name, position, percentage=False):
    """
    Add a line chart for vintage performance.
    
    Args:
        ws: Worksheet object
        vintage_data: List of dicts with vintage performance data
        title: Chart title
        metric_name: Name of the metric to chart
        position: Chart position (cell reference)
        percentage: Whether the metric is a percentage
    """
    # Add a data range for the chart
    row_offset = ws.max_row + 2
    ws.cell(row=row_offset, column=1, value="Vintage")
    ws.cell(row=row_offset, column=2, value=metric_name)
    
    # Add data rows
    for i, v_data in enumerate(vintage_data, 1):
        ws.cell(row=row_offset+i, column=1, value=v_data['Vintage'])
        
        if not np.isnan(v_data[metric_name]):
            ws.cell(row=row_offset+i, column=2, value=v_data[metric_name])
            if percentage:
                ws.cell(row=row_offset+i, column=2).number_format = '0.00%'
            else:
                ws.cell(row=row_offset+i, column=2).number_format = '0.00x'
    
    # Create chart
    chart = LineChart()
    chart.title = title
    chart.style = 2  # Choose the style you want
    chart.x_axis.title = "Vintage"
    chart.y_axis.title = metric_name
    
    # Configure data
    data = Reference(ws, min_col=2, min_row=row_offset, max_row=row_offset+len(vintage_data))
    chart.add_data(data, titles_from_data=True)
    
    # Configure categories (vintage)
    cats = Reference(ws, min_col=1, min_row=row_offset+1, max_row=row_offset+len(vintage_data))
    chart.set_categories(cats)
    
    # Add markers to the lines
    s = chart.series[0]
    s.marker = Marker(symbol="circle")
    s.marker.size = 7
    
    # Add the chart to the worksheet
    ws.add_chart(chart, position)
    
    return chart


if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    from portfolio_simulator import Portfolio, AllocationPolicy, FeeSchedule
    
    # Load data
    deals_df, payments_df = load_data()
    
    # Create and simulate a portfolio
    portfolio = Portfolio(
        name="Test Portfolio",
        selection_criteria={
            'product_types': ['RBF'],
            'deal_size_range': (100000, 500000),
            'vintage_range': ('2023-01-01', '2023-12-31')
        },
        allocation_policy=AllocationPolicy(
            percentage=0.25, 
            min_amount=100000, 
            max_amount=1500000
        ),
        fee_schedule=FeeSchedule(
            percentage=0.02,
            apply_after_principal=True
        )
    )
    
    # Simulate the portfolio
    portfolio.simulate(deals_df, payments_df)
    
    # Export to Excel
    export_portfolio_to_excel(portfolio, "test_portfolio_export.xlsx", verbose=True)