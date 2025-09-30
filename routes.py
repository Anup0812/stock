from flask import Blueprint, render_template, jsonify, request
from utils.data_loader import load_server_data, process_server_data, get_server_stats

windows_bp = Blueprint('windows', __name__, url_prefix='/team/windows')

def get_windows_servers():
    """Get Windows servers with specific filtering logic"""
    data = load_server_data()
    servers = process_server_data(data, vendor_filter=['Windows'])
    return servers

@windows_bp.route('/')
def dashboard():
    """Windows team dashboard"""
    servers = get_windows_servers()
    stats = get_server_stats(servers)
    return render_template('teams/windows/dashboard.html',
                         team_name='Windows',
                         stats=stats)

@windows_bp.route('/server_health')
def server_health():
    """Windows server health page"""
    servers = get_windows_servers()
    stats = get_server_stats(servers)

    # Handle filtering from query parameters
    filter_type = request.args.get('filter')
    if filter_type:
        if filter_type == 'critical':
            servers = [s for s in servers if s['Status'] != 1]
        elif filter_type == 'high_cpu':
            servers = [s for s in servers if s['CPULoad'] > 50]
        elif filter_type == 'high_memory':
            servers = [s for s in servers if s['MemoryUsagePercent'] > 80]
        elif filter_type == 'high_disk':
            servers = [s for s in servers if s['MaxDiskUsage'] > 70]

    return render_template('teams/windows/server_health.html',
                         servers=servers,
                         team_name='Windows',
                         stats=stats,
                         current_filter=filter_type)

@windows_bp.route('/api/server_data')
def api_server_data():
    """API endpoint for Windows server data"""
    servers = get_windows_servers()
    return jsonify(servers)

@windows_bp.route('/service_health')
def service_health():
    """Windows service health check placeholder"""
    return render_template('placeholder.html',
                         module='Service Health Check',
                         team_name='Windows')

@windows_bp.route('/vmware_health')
def vmware_health():
    """Windows VMware health check placeholder"""
    return render_template('placeholder.html',
                         module='VMware Health Check',
                         team_name='Windows')

@windows_bp.route('/reports')
def reports():
    """Windows reports placeholder"""
    return render_template('placeholder.html',
                         module='Reports',
                         team_name='Windows')

from flask import send_file
import io
import openpyxl


@windows_bp.route('/export_report')
def export_report():
    """Export Windows server health report to Excel with two worksheets"""
    servers = get_windows_servers()

    # Create an Excel workbook
    wb = openpyxl.Workbook()

    # ===== First Sheet: All Servers =====
    ws1 = wb.active
    ws1.title = "All Servers"

    # Define headers
    headers = ["Server Name", "Status", "Operating System", "IP Address", "CPU Load (%)",
               "Memory Usage (%)", "Max Disk Usage (%)", "Total Memory (GB)"]
    ws1.append(headers)

    # Fill rows with all server data
    for s in servers:
        ws1.append([
            s["NodeName"],
            "Online" if s["Status"] == 1 else "Offline",
            s["OperatingSystem"],
            s["IPAddress"],
            s["CPULoad"],
            s["MemoryUsagePercent"],
            s["MaxDiskUsage"],
            s["TotalMemorySize"]
        ])

    # ===== Second Sheet: Critical Issues =====
    ws2 = wb.create_sheet(title="Critical Issues")

    # Add headers with Issue Type column
    critical_headers = ["Server Name", "Status", "Operating System", "IP Address",
                        "CPU Load (%)", "Memory Usage (%)", "Max Disk Usage (%)",
                        "Total Memory (GB)", "Issue Type"]
    ws2.append(critical_headers)

    # Filter and add servers with critical issues
    for s in servers:
        issues = []

        # Check for various critical conditions
        if s["Status"] != 1:
            issues.append("Offline")
        if s["CPULoad"] > 50:
            issues.append("High CPU")
        if s["MemoryUsagePercent"] > 80:
            issues.append("High Memory")
        if s["MaxDiskUsage"] > 70:
            issues.append("High Disk")

        # Only add servers that have at least one issue
        if issues:
            ws2.append([
                s["NodeName"],
                "Online" if s["Status"] == 1 else "Offline",
                s["OperatingSystem"],
                s["IPAddress"],
                s["CPULoad"],
                s["MemoryUsagePercent"],
                s["MaxDiskUsage"],
                s["TotalMemorySize"],
                ", ".join(issues)  # Combine all issues into one column
            ])

    # Save workbook to a bytes buffer
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="windows_server_health.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )