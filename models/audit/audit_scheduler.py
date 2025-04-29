from PyQt5.QtWidgets import QInputDialog, QMessageBox
from PyQt5.QtCore import QTimer
#from PyQt5.QtGui import 
from datetime import datetime, timedelta
import re
import QtCore

class AuditScheduler:
    def __init__(self, parent, output_callback, status_callback):
        self.parent = parent
        self.output_callback = output_callback
        self.status_callback = status_callback


    def start_audit_now(self):
        """Immediately start the audit process"""
        from models.auditor import CodeAuditor
        self.auditor = CodeAuditor("src/")
        issues = self.auditor.run_audit()
        new_issues = self.auditor.log_issues(issues)
        self.show_status_message(f"Audit completed. Found {len(new_issues)} issues.", 5000)
        self.output_area.append(f"<font color='gold'>[AUDIT]</font> Scan completed with {len(new_issues)} findings.<br>")

    def schedule_audit(self):
        """Schedule audit for later time"""
        options = ["Specific Time (HH:MM:SS)", "Countdown (MMM:SS)"]
        choice, ok = QInputDialog.getItem(
            self, "Schedule Audit",
            "Select scheduling method:", options, 0, False
        )
        
        if not ok:
            return

        if choice == options[0]:  # Specific time
            time_str, ok = QInputDialog.getText(
                self, "Enter Time",
                "Enter audit time (24h format HH:MM:SS):",
                text=datetime.now().strftime("%H:%M:%S")
            )
            if ok and self.validate_time_format(time_str):
                self.schedule_at_specific_time(time_str)
        
        else:  # Countdown
            time_str, ok = QInputDialog.getText(
                self, "Enter Duration",
                "Enter countdown (MMM:SS):",
                text="005:00"  # Default 5 minutes
            )
            if ok and self.validate_countdown_format(time_str):
                self.schedule_countdown(time_str)

    def validate_time_format(self, time_str):
        """Validate HH:MM:SS format"""
        if not re.match(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$', time_str):
            QMessageBox.warning(self, "Invalid Format", "Please use HH:MM:SS 24-hour format")
            return False
        return True

    def validate_countdown_format(self, time_str):
        """Validate MMM:SS format"""
        if not re.match(r'^\d{3}:[0-5][0-9]$', time_str):
            QMessageBox.warning(self, "Invalid Format", "Please use MMM:SS format (e.g. 132:45)")
            return False
        return True

    def schedule_at_specific_time(self, time_str):
        """Schedule audit for specific clock time"""
        now = datetime.now()
        target_time = datetime.strptime(time_str, "%H:%M:%S").time()
        target_datetime = datetime.combine(now.date(), target_time)
        
        if target_datetime < now:
            target_datetime += timedelta(days=1)  # Schedule for next day
        
        delay_seconds = (target_datetime - now).total_seconds()
        self.show_status_message(f"Audit scheduled for {time_str}", 3000)
        QtCore.QTimer.singleShot(int(delay_seconds * 1000), self.start_audit_now)

    def schedule_countdown(self, time_str):
        """Schedule audit after countdown duration"""
        minutes, seconds = map(int, time_str.split(':'))
        total_seconds = minutes * 60 + seconds
        self.show_status_message(f"Audit scheduled in {minutes}m {seconds}s", 3000)
        QtCore.QTimer.singleShot(total_seconds * 1000, self.start_audit_now)
