import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, DatetimeTickFormatter, LinearAxis, Range1d

# Read the Excel file
def read_excel_file(file_path):
    try:
        data = pd.read_excel(file_path)
        data = data.rename(columns={'Time(DD/MM/YYYY h:mm:ss A)': 'Time'})
        data['Time'] = pd.to_datetime(data['Time'], format='%d/%m/%Y %I:%M:%S %p')
        return data
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return None

# Plot the data
def plot_data(data):
    source = ColumnDataSource(data)

    p = figure(title="Aranet4 PRO data", x_axis_label='Time', y_axis_label='CO2 (ppm)', x_axis_type='datetime', width=1600, height=900)

    co2_line = p.line('Time', 'Carbon dioxide(ppm)', line_color="blue", source=source)
    hover_co2 = HoverTool(tooltips=[("Time", "@Time{%F %H:%M}"), ("CO2 (ppm)", "@{Carbon dioxide(ppm)}")], formatters={'@Time': 'datetime'}, renderers=[co2_line])
    p.add_tools(hover_co2)

    p.extra_y_ranges['temp'] = Range1d(start=min(data['Temperature(°C)'])*0.9, end=max(data['Temperature(°C)'])*1.1)
    p.add_layout(LinearAxis(y_range_name='temp', axis_label='Temperature (°C)'), 'right')
    temp_line = p.line('Time', 'Temperature(°C)', line_color="red", y_range_name='temp', source=source)
    hover_temp = HoverTool(tooltips=[("Time", "@Time{%F %H:%M}"), ("Temperature (°C)", "@{Temperature(°C)}")], formatters={'@Time': 'datetime'}, renderers=[temp_line])
    p.add_tools(hover_temp)

    p.extra_y_ranges['humidity'] = Range1d(start=min(data['Relative humidity(%)'])*0.9, end=max(data['Relative humidity(%)'])*1.1)
    p.add_layout(LinearAxis(y_range_name='humidity', axis_label='Relative Humidity (%)'), 'right')
    humidity_line = p.line('Time', 'Relative humidity(%)', line_color="green", y_range_name='humidity', source=source)
    hover_humidity = HoverTool(tooltips=[("Time", "@Time{%F %H:%M}"), ("Relative Humidity (%)", "@{Relative humidity(%)}")], formatters={'@Time': 'datetime'}, renderers=[humidity_line])
    p.add_tools(hover_humidity)

    p.extra_y_ranges['pressure'] = Range1d(start=min(data['Atmospheric pressure(hPa)'])*0.9, end=max(data['Atmospheric pressure(hPa)'])*1.1)
    p.add_layout(LinearAxis(y_range_name='pressure', axis_label='Atmospheric Pressure (hPa)'), 'right')
    pressure_line = p.line('Time', 'Atmospheric pressure(hPa)', line_color="purple", y_range_name='pressure', source=source)
    hover_pressure = HoverTool(tooltips=[("Time", "@Time{%F %H:%M}"), ("Atmospheric Pressure (hPa)", "@{Atmospheric pressure(hPa)}")], formatters={'@Time': 'datetime'}, renderers=[pressure_line])
    p.add_tools(hover_pressure)

    p.xaxis.formatter = DatetimeTickFormatter(
        minutes="%Y-%m-%d %H:%M",
        hours="%Y-%m-%d %H:%M",
        days="%Y-%m-%d",
        months="%Y-%m",
        years="%Y"
    )

    show(p)

# Main function
def main():
    file_path = r"C:\Users\nml\OneDrive - NIST\Documents\NIST\EPA_Legionella\Aranet4_PRO\Bedroom_2025-12-19T13_37_21-0500_today.xlsx"
    data = read_excel_file(file_path)
    if data is not None:
        plot_data(data)

if __name__ == "__main__":
    main()