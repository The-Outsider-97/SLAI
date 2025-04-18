import time
import os, sys


def exponential_backoff(retries):
    delay = 2 ** retries
    time.sleep(delay)
    return delay

class SecurityFeatures:
    def detect_captcha(driver):
        """Check for CAPTCHA challenges in page content"""
        captcha_indicators = [
            "captcha",
            "CAPTCHA",
            "robot check",
            "verify you are human"
        ]
        page_source = driver.page_source.lower()
        return any(indicator in page_source for indicator in captcha_indicators)
