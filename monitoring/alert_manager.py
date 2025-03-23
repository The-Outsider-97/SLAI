import logging
import smtplib
from email.mime.text import MIMEText
from log_handler import log_event

def send_alert(subject, body, to_email):
    from_email = "your_email@example.com"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP("smtp.example.com", 587) as server:
            server.starttls()
            server.login(from_email, "your_password")
            server.sendmail(from_email, to_email, msg.as_string())
        log_event(f"Alert sent to {to_email}: {subject}")
    except Exception as e:
        log_event(f"Failed to send alert: {e}", level="error")

if __name__ == "__main__":
    send_alert("Test Alert", "This is a test alert from SLAI!", "receiver@example.com")
