import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import random
import datetime
import os
import webbrowser
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
import numpy as np
import time
import queue

# Dictionary of patients
patients = {i: f"Patient_{i+1}" for i in range(100)}

# Class for monitoring data and displaying graphs
class DataMonitor(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.selected_patient = tk.StringVar(value="Patient_1")
        self.duration = tk.StringVar(value="30")

        self.patient_selector = ttk.Combobox(self, textvariable=self.selected_patient, values=list(patients.values()))
        self.patient_selector.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.patient_selector.bind("<<ComboboxSelected>>", self.on_patient_change)

        self.patient_label = tk.Label(self, text=f"Selected Patient: {self.selected_patient.get()}", bg="#282c34", fg="#ffffff", font=("Helvetica", 12))
        self.patient_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.duration_entry = tk.Entry(self, textvariable=self.duration)
        self.duration_entry.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)

        self.timestamps = {i: [] for i in range(100)}
        self.temperatures = {i: [] for i in range(100)}
        self.heart_rates = {i: [] for i in range(100)}
        self.oxygen_levels = {i: [] for i in range(100)}
        self.alerts = {i: [] for i in range(100)}

        self.forecast_temp = {i: [] for i in range(100)}
        self.forecast_hr = {i: [] for i in range(100)}
        self.forecast_ox = {i: [] for i in range(100)}

        self.line_temp, = self.axs[0].plot([], label="Temperature")
        self.line_hr, = self.axs[1].plot([], label="Heart Rate")
        self.line_ox, = self.axs[2].plot([], label="Oxygen Level")

        self.forecast_line_temp, = self.axs[0].plot([], label="Forecast Temp", linestyle='--')
        self.forecast_line_hr, = self.axs[1].plot([], label="Forecast HR", linestyle='--')
        self.forecast_line_ox, = self.axs[2].plot([], label="Forecast O2", linestyle='--')

        self.axs[0].set_title('Temperature')
        self.axs[1].set_title('Heart Rate')
        self.axs[2].set_title('Oxygen Level')

        for ax in self.axs:
            ax.set_xlim(0, 60)
            ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.draw()

        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

        self.update_interval = 1000  # 1 second
        self.update_graph()

    def on_patient_change(self, event):
        self.patient_label.config(text=f"Selected Patient: {self.selected_patient.get()}")
        self.clear_data()
        self.update_graph()

    def clear_data(self):
        self.line_temp.set_data([], [])
        self.line_hr.set_data([], [])
        self.line_ox.set_data([], [])
        self.forecast_line_temp.set_data([], [])
        self.forecast_line_hr.set_data([], [])
        self.forecast_line_ox.set_data([], [])
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw()

    def update_graph(self):
        selected_patient_id = list(patients.values()).index(self.selected_patient.get())
        temp_data = self.temperatures[selected_patient_id]
        hr_data = self.heart_rates[selected_patient_id]
        ox_data = self.oxygen_levels[selected_patient_id]
        alert_data = self.alerts[selected_patient_id]
        timestamps = self.timestamps[selected_patient_id]

        forecast_temp = self.forecast_temp[selected_patient_id]
        forecast_hr = self.forecast_hr[selected_patient_id]
        forecast_ox = self.forecast_ox[selected_patient_id]

        if len(temp_data) > 60:
            temp_data = temp_data[-60:]
            hr_data = hr_data[-60:]
            ox_data = ox_data[-60:]
            alert_data = alert_data[-60:]
            timestamps = timestamps[-60:]

        self.line_temp.set_ydata(temp_data)
        self.line_temp.set_xdata(range(len(temp_data)))

        self.line_hr.set_ydata(hr_data)
        self.line_hr.set_xdata(range(len(hr_data)))

        self.line_ox.set_ydata(ox_data)
        self.line_ox.set_xdata(range(len(ox_data)))

        self.forecast_line_temp.set_ydata(forecast_temp)
        self.forecast_line_temp.set_xdata(range(len(temp_data), len(temp_data) + len(forecast_temp)))

        self.forecast_line_hr.set_ydata(forecast_hr)
        self.forecast_line_hr.set_xdata(range(len(hr_data), len(hr_data) + len(forecast_hr)))

        self.forecast_line_ox.set_ydata(forecast_ox)
        self.forecast_line_ox.set_xdata(range(len(ox_data), len(ox_data) + len(forecast_ox)))

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        self.axs[0].clear()
        self.axs[1].clear()
        self.axs[2].clear()

        self.axs[0].plot(range(len(temp_data)), temp_data, label="Temperature", color='blue')
        self.axs[0].scatter(range(len(temp_data)), temp_data, color=['red' if alert else 'blue' for alert in alert_data])
        self.axs[0].plot(range(len(temp_data), len(temp_data) + len(forecast_temp)), forecast_temp, linestyle='--', color='green', label="Forecast Temp")
        self.axs[0].set_title('Temperature')
        self.axs[0].legend()

        self.axs[1].plot(range(len(hr_data)), hr_data, label="Heart Rate", color='blue')
        self.axs[1].scatter(range(len(hr_data)), hr_data, color=['red' if alert else 'blue' for alert in alert_data])
        self.axs[1].plot(range(len(hr_data), len(hr_data) + len(forecast_hr)), forecast_hr, linestyle='--', color='green', label="Forecast HR")
        self.axs[1].set_title('Heart Rate')
        self.axs[1].legend()

        self.axs[2].plot(range(len(ox_data)), ox_data, label="Oxygen Level", color='blue')
        self.axs[2].scatter(range(len(ox_data)), ox_data, color=['red' if alert else 'blue' for alert in alert_data])
        self.axs[2].plot(range(len(ox_data), len(ox_data) + len(forecast_ox)), forecast_ox, linestyle='--', color='green', label="Forecast O2")
        self.axs[2].set_title('Oxygen Level')
        self.axs[2].legend()

        self.canvas.draw()

        self.after(self.update_interval, self.update_graph)

    def update_data(self, producer_id, timestamp, temperature, heart_rate, oxygen_level, alert_detected):
        self.timestamps[producer_id].append(timestamp)
        self.temperatures[producer_id].append(temperature)
        self.heart_rates[producer_id].append(heart_rate)
        self.oxygen_levels[producer_id].append(oxygen_level)
        self.alerts[producer_id].append(alert_detected)

        if len(self.timestamps[producer_id]) > 100:
            self.timestamps[producer_id] = self.timestamps[producer_id][-100:]
            self.temperatures[producer_id] = self.temperatures[producer_id][-100:]
            self.heart_rates[producer_id] = self.heart_rates[producer_id][-100:]
            self.oxygen_levels[producer_id] = self.oxygen_levels[producer_id][-100:]
            self.alerts[producer_id] = self.alerts[producer_id][-100:]

    def update_forecasts(self, patient_id, forecast_temp, forecast_hr, forecast_ox):
        self.forecast_temp[patient_id] = forecast_temp
        self.forecast_hr[patient_id] = forecast_hr
        self.forecast_ox[patient_id] = forecast_ox

    def get_data(self, patient_id):
        return (self.timestamps[patient_id], self.temperatures[patient_id], self.heart_rates[patient_id], self.oxygen_levels[patient_id])

    def on_hover(self, event):
        if event.inaxes in self.axs:
            for line in [self.line_temp, self.line_hr, self.line_ox]:
                cont, ind = line.contains(event)
                if cont:
                    idx = ind["ind"][0]
                    selected_patient_id = list(patients.values()).index(self.selected_patient.get())
                    timestamp = self.timestamps[selected_patient_id][idx]
                    temp = self.temperatures[selected_patient_id][idx]
                    hr = self.heart_rates[selected_patient_id][idx]
                    ox = self.oxygen_levels[selected_patient_id][idx]
                    alert = self.alerts[selected_patient_id][idx]
                    alert_status = "Alert" if alert else "Normal"

                    details = f"Time: {timestamp}\nTemp: {temp}\nHeart Rate: {hr}\nOxygen: {ox}\nStatus: {alert_status}"
                    self.show_details(details)
                    break

    def show_details(self, details):
        messagebox.showinfo("Data Point Details", details)

    def show_history(self):
        selected_patient_id = list(patients.values()).index(self.selected_patient.get())
        duration = int(self.duration.get())
        temp_data = self.temperatures[selected_patient_id][-duration:]
        hr_data = self.heart_rates[selected_patient_id][-duration:]
        ox_data = self.oxygen_levels[selected_patient_id][-duration:]
        alert_data = self.alerts[selected_patient_id][-duration:]
        timestamps = self.timestamps[selected_patient_id][-duration:]

        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        fig.tight_layout(pad=3.0)

        axs[0].plot(temp_data, label="Temperature", color='blue')
        axs[0].scatter(range(len(temp_data)), temp_data, color=['red' if alert else 'blue' for alert in alert_data])
        axs[0].set_title('Temperature')
        axs[0].legend()

        axs[1].plot(hr_data, label="Heart Rate", color='blue')
        axs[1].scatter(range(len(hr_data)), hr_data, color=['red' if alert else 'blue' for alert in alert_data])
        axs[1].set_title('Heart Rate')
        axs[1].legend()

        axs[2].plot(ox_data, label="Oxygen Level", color='blue')
        axs[2].scatter(range(len(ox_data)), ox_data, color=['red' if alert else 'blue' for alert in alert_data])
        axs[2].set_title('Oxygen Level')
        axs[2].legend()

        for ax in axs:
            ax.set_xticklabels([])

        self.history_canvas.figure = fig
        self.history_canvas.draw()

        self.history_canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes:
            for line in event.inaxes.lines:
                cont, ind = line.contains(event)
                if cont:
                    idx = ind["ind"][0]
                    selected_patient_id = list(patients.values()).index(self.selected_patient.get())
                    temp = self.temperatures[selected_patient_id][idx]
                    hr = self.heart_rates[selected_patient_id][idx]
                    ox = self.oxygen_levels[selected_patient_id][idx]
                    timestamp = self.timestamps[selected_patient_id][idx]
                    alert = self.alerts[selected_patient_id][idx]
                    alert_status = "Alert" if alert else "Normal"

                    details = f"Time: {timestamp}\nTemp: {temp}\nHeart Rate: {hr}\nOxygen: {ox}\nStatus: {alert_status}"
                    self.show_details(details)
                    break

    def open_history_window(self):
        history_window = tk.Toplevel(self)
        history_window.title("History")
        history_window.geometry("800x600")

        self.history_canvas = FigureCanvasTkAgg(plt.Figure(), master=history_window)
        self.history_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.show_history()

# Class for the alert window
class AlertWindow(tk.Toplevel):
    def __init__(self, master):
        tk.Toplevel.__init__(self, master)
        self.title("Alert Table")
        self.geometry("600x400")
        self.configure(bg="#282c34")
        
        self.alert_table = ttk.Treeview(self, columns=("Patient", "Alert"), show='headings', selectmode="browse")
        self.alert_table.heading("Patient", text="Patient")
        self.alert_table.heading("Alert", text="Alert")
        self.alert_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.alert_table.column("Patient", width=150)
        self.alert_table.column("Alert", width=450)

        self.alert_table.tag_configure('oddrow', background='#1c1f24')
        self.alert_table.tag_configure('evenrow', background='#2c313a')
        
        self.alert_table.bind("<Configure>", self.adjust_row_colors)

    def adjust_row_colors(self, event):
        for i, item in enumerate(self.alert_table.get_children()):
            if i % 2 == 0:
                self.alert_table.item(item, tags=('evenrow',))
            else:
                self.alert_table.item(item, tags=('oddrow',))

class Application(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Real-Time Health Parameter Monitoring")
        self.geometry("1200x800")
        self.configure(bg="#282c34")
        
        main_frame = tk.Frame(self, bg="#282c34")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.log_widget = scrolledtext.ScrolledText(main_frame, width=80, height=15, bg="#1c1f24", fg="#ffffff", font=("Helvetica", 10))
        self.log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.log_widget.tag_config('info', foreground='lightblue')
        self.log_widget.tag_config('alert', foreground='red', font=("Helvetica", 10, "bold"))
        self.log_widget.tag_config('forecast', foreground='green', font=("Helvetica", 10, "italic"))
        self.log_widget.tag_config('error', foreground='red', font=("Helvetica", 10, "italic"))

        self.start_button = tk.Button(main_frame, text="Start", command=self.start_all, bg="#98c379", fg="#ffffff", font=("Helvetica", 12, "bold"))
        self.start_button.pack(padx=10, pady=10)

        self.stop_button = tk.Button(main_frame, text="Stop", command=self.stop_all, bg="#e06c75", fg="#ffffff", font=("Helvetica", 12, "bold"))
        self.stop_button.pack(padx=10, pady=10)

        self.view_graph_button = tk.Button(main_frame, text="View History", command=self.open_history_window, bg="#61afef", fg="#ffffff", font=("Helvetica", 12, "bold"))
        self.view_graph_button.pack(padx=10, pady=10)

        self.generate_report_button = tk.Button(main_frame, text="Generate Report", command=self.generate_alert_report, bg="#d19a66", fg="#ffffff", font=("Helvetica", 12, "bold"))
        self.generate_report_button.pack(padx=10, pady=10)

        self.open_report_button = tk.Button(main_frame, text="Open Report", command=self.open_report, bg="#c678dd", fg="#ffffff", font=("Helvetica", 12, "bold"))
        self.open_report_button.pack(padx=10, pady=10)

        self.alert_log = []
        self.report_file = "all_patients_alert_report.pdf"

        self.data_monitor = DataMonitor(main_frame)
        self.data_monitor.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.alert_window = AlertWindow(self)

        self.running = False
        self.data_queue = queue.Queue()

    def open_history_window(self):
        self.data_monitor.open_history_window()

    def start_all(self):
        self.running = True
        self.alert_log = []
        start_time = time.time()
        self.run_sequentially()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.log_widget.insert(tk.END, f"Total execution time: {elapsed_time:.2f} seconds\n", 'info')

    def stop_all(self):
        self.running = False
        self.generate_alert_report()

    def run_sequentially(self):
        while self.running:
            for producer_id in range(100):
                if not self.running:
                    break
                self.collect_data(producer_id)
            for consumer_id in range(10):
                if not self.running:
                    break
                self.process_data(consumer_id)
            self.make_forecasts()
            self.update_gui()
            time.sleep(1)

    def collect_data(self, producer_id):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temperature = random.uniform(36.0, 39.0)
        heart_rate = random.randint(60, 120)
        oxygen_level = random.uniform(90.0, 100.0)
        blood_pressure = (random.randint(90, 140), random.randint(60, 90))
        item = f"{timestamp}, Temp: {temperature:.2f}, Heart Rate: {heart_rate}, O2: {oxygen_level:.2f}, BP: {blood_pressure[0]}/{blood_pressure[1]}"
        self.log_widget.insert(tk.END, f"Producer {producer_id} ({patients[producer_id]}) added: {item}\n", 'info')
        self.log_widget.yview(tk.END)
        print(f"Producer {producer_id} ({patients[producer_id]}) added: {item}")
        self.data_queue.put((producer_id, item))

    def process_data(self, consumer_id):
        try:
            producer_id, item = self.data_queue.get(timeout=1)
            self.log_widget.insert(tk.END, f"Consumer {consumer_id} took from Producer {producer_id} ({patients[producer_id]}): {item}\n", 'info')
            self.log_widget.yview(tk.END)
            print(f"Consumer {consumer_id} took from Producer {producer_id} ({patients[producer_id]}): {item}")
            self.process_item(consumer_id, producer_id, item)
        except queue.Empty:
            pass

    def process_item(self, consumer_id, producer_id, item):
        data = item.split(", ")
        timestamp = data[0]
        temp = float(data[1].split(": ")[1])
        heart_rate = int(data[2].split(": ")[1])
        oxygen_level = float(data[3].split(": ")[1])
        bp = tuple(map(int, data[4].split(": ")[1].split("/")))

        alert = ""
        if temp > 37.5:
            alert += "Fever detected! "
        if heart_rate > 100:
            alert += "Tachycardia detected! "
        if oxygen_level < 95.0:
            alert += "Hypoxia detected! "
        if bp[0] > 130 or bp[1] > 85:
            alert += "Hypertension detected! "

        alert_detected = bool(alert)
        self.data_monitor.update_data(producer_id, timestamp, temp, heart_rate, oxygen_level, alert_detected)

        if alert_detected:
            alert_message = f"ALERT by Consumer {consumer_id} for {patients[producer_id]}: {alert}"
            self.log_widget.insert(tk.END, alert_message + '\n', 'alert')
            self.log_widget.yview(tk.END)
            print(alert_message)
            alert_id = self.alert_window.alert_table.insert('', 'end', values=(patients[producer_id], alert))
            self.alert_window.alert_table.see(alert_id)
            self.alert_log.append((patients[producer_id], timestamp, temp, heart_rate, oxygen_level, bp, alert))

    def make_forecasts(self):
        for patient_id in range(100):
            timestamps, temperatures, heart_rates, oxygen_levels = self.data_monitor.get_data(patient_id)

            if len(timestamps) >= 2:
                try:
                    future_timestamps = np.array(range(len(timestamps), len(timestamps) + 10)).reshape(-1, 1)

                    temp_model = LinearRegression().fit(np.array(range(len(timestamps))).reshape(-1, 1), temperatures)
                    temp_forecast = temp_model.predict(future_timestamps)

                    hr_model = LinearRegression().fit(np.array(range(len(timestamps))).reshape(-1, 1), heart_rates)
                    hr_forecast = hr_model.predict(future_timestamps)

                    ox_model = LinearRegression().fit(np.array(range(len(timestamps))).reshape(-1, 1), oxygen_levels)
                    ox_forecast = ox_model.predict(future_timestamps)

                    self.data_monitor.update_forecasts(patient_id, temp_forecast, hr_forecast, ox_forecast)

                    forecast_message = f"Forecast for {patients[patient_id]} - Temp: {temp_forecast[-1]:.2f}, HR: {hr_forecast[-1]:.2f}, O2: {ox_forecast[-1]:.2f}"
                    self.log_widget.insert(tk.END, forecast_message + '\n', 'forecast')
                    self.log_widget.yview(tk.END)
                    print(forecast_message)
                except Exception as e:
                    self.log_widget.insert(tk.END, f"Error forecasting for {patients[patient_id]}: {str(e)}\n", 'error')
                    self.log_widget.yview(tk.END)
                    print(f"Error forecasting for {patients[patient_id]}: {str(e)}")

    def update_gui(self):
        self.data_monitor.update_graph()
        self.update()

    def generate_alert_report(self):
        alerts_by_patient = {}
        for alert in self.alert_log:
            patient_name = alert[0]
            if patient_name not in alerts_by_patient:
                alerts_by_patient[patient_name] = []
            alerts_by_patient[patient_name].append(alert)

        doc = SimpleDocTemplate(self.report_file, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='TableHeader', alignment=TA_CENTER, fontSize=10, fontName='Helvetica-Bold', textColor=colors.whitesmoke, backColor=colors.grey, padding=3))
        styles.add(ParagraphStyle(name='TableCell', alignment=TA_CENTER, fontSize=8, fontName='Helvetica', padding=3))

        title = Paragraph("Comprehensive Alert Report", styles['Title'])
        elements.append(title)
        elements.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 12))

        for patient_name, alerts in alerts_by_patient.items():
            elements.append(Paragraph(f"Alert Report for {patient_name}", styles['Heading2']))
            elements.append(Spacer(1, 12))

            table_data = [
                [Paragraph("Timestamp", styles['TableHeader']),
                 Paragraph("Temperature", styles['TableHeader']),
                 Paragraph("Heart Rate", styles['TableHeader']),
                 Paragraph("Oxygen Level", styles['TableHeader']),
                 Paragraph("Blood Pressure", styles['TableHeader']),
                 Paragraph("Alert", styles['TableHeader'])]]

            for alert in alerts:
                row = [Paragraph(alert[1], styles['TableCell']),
                       Paragraph(str(alert[2]), styles['TableCell']),
                       Paragraph(str(alert[3]), styles['TableCell']),
                       Paragraph(str(alert[4]), styles['TableCell']),
                       Paragraph(f"{alert[5][0]}/{alert[5][1]}", styles['TableCell']),
                       Paragraph(alert[6], styles['TableCell'])]
                table_data.append(row)

            table = Table(table_data, colWidths=[doc.width/6.0]*6)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 😎,
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 6)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 24))

        doc.build(elements)

        self.log_widget.insert(tk.END, f"Comprehensive report generated: {self.report_file}\n", 'info')
        print(f"Comprehensive report generated: {self.report_file}")

    def open_report(self):
        if os.path.exists(self.report_file):
            webbrowser.get('firefox').open_new_tab(self.report_file)
            self.log_widget.insert(tk.END, f"Opened report in Firefox: {self.report_file}\n", 'info')
            print(f"Opened report in Firefox: {self.report_file}")
        else:
            self.log_widget.insert(tk.END, f"Report file does not exist: {self.report_file}\n", 'error')
            print(f"Report file does not exist: {self.report_file}")

if name == "main":
    app = Application()
    app.mainloop()
