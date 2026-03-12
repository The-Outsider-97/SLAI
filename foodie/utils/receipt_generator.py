
import os
from pathlib import Path
from fpdf import FPDF
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def generate_order_receipt(order_data, output_dir="static/order_receipts", format="pdf"):
    """Generate order receipt in specified format"""
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create unique filename
    filename = f"business_order_{order_data['order_id']}"
    filepath = os.path.join(output_dir, filename)
    
    if format == "pdf":
        return _generate_pdf_receipt(order_data, filepath + ".pdf")
    else:
        return _generate_jpg_receipt(order_data, filepath + ".jpg")

def _generate_pdf_receipt(order_data, filepath):
    """Generate PDF receipt"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Business Order Receipt", ln=1, align="C")
    pdf.set_font("Arial", size=12)
    
    # Add order details
    pdf.cell(200, 10, txt=f"ID number: {order_data['order_id']}", ln=1)
    pdf.cell(200, 10, txt=f"Order time/date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    
    # Add order specificity
    pdf.cell(200, 10, txt="Order specificity:", ln=1)
    
    # Company info
    pdf.cell(200, 10, txt="Company Info:", ln=1)
    company = order_data["details"]["company_info"]
    pdf.cell(200, 10, txt=f"  Name: {company['name']}", ln=1)
    pdf.cell(200, 10, txt=f"  Address: {company['address']}", ln=1)
    pdf.cell(200, 10, txt=f"  Registration: {company['registration_number']}", ln=1)
    
    # Contact info
    pdf.cell(200, 10, txt="Contact Info:", ln=1)
    contact = order_data["details"]["contact_info"]
    pdf.cell(200, 10, txt=f"  Name: {contact['name']}", ln=1)
    pdf.cell(200, 10, txt=f"  Email: {contact['email']}", ln=1)
    pdf.cell(200, 10, txt=f"  Phone: {contact['phone']}", ln=1)
    
    # Requirements
    pdf.cell(200, 10, txt="Requirements:", ln=1)
    req = order_data["details"]["requirements"]
    pdf.cell(200, 10, txt=f"  Employee Count: {req['employee_count']}", ln=1)
    pdf.cell(200, 10, txt=f"  Frequency: {req['frequency']}", ln=1)
    pdf.cell(200, 10, txt=f"  Cuisine Preferences: {', '.join(req['cuisine_preferences'])}", ln=1)
    pdf.cell(200, 10, txt=f"  Special Requests: {req['special_requests']}", ln=1)
    pdf.cell(200, 10, txt=f"  Arrival Time: {req['arrival_time']}", ln=1)
    
    # Save PDF
    pdf.output(filepath)
    return filepath

def _generate_jpg_receipt(order_data, filepath):
    """Generate JPG receipt image"""
    # Create image with white background
    img = Image.new('RGB', (800, 1200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y_position = 20
    
    # Draw title
    d.text((20, y_position), "Business Order Receipt", font=font, fill=(0, 0, 0))
    y_position += 30
    
    # Draw order details
    d.text((20, y_position), f"ID number: {order_data['order_id']}", font=font, fill=(0, 0, 0))
    y_position += 20
    d.text((20, y_position), f"Order time/date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}", font=font, fill=(0, 0, 0))
    y_position += 30
    
    # Draw order specificity
    d.text((20, y_position), "Order specificity:", font=font, fill=(0, 0, 0))
    y_position += 20
    
    # Company info
    d.text((40, y_position), "Company Info:", font=font, fill=(0, 0, 0))
    y_position += 20
    company = order_data["details"]["company_info"]
    d.text((60, y_position), f"Name: {company['name']}", font=font, fill=(0, 0, 0))
    y_position += 20
    d.text((60, y_position), f"Address: {company['address']}", font=font, fill=(0, 0, 0))
    y_position += 20
    d.text((60, y_position), f"Registration: {company['registration_number']}", font=font, fill=(0, 0, 0))
    y_position += 30
    
    # Contact info
    d.text((40, y_position), "Contact Info:", font=font, fill=(0, 0, 0))
    y_position += 20
    contact = order_data["details"]["contact_info"]
    d.text((60, y_position), f"Name: {contact['name']}", font=font, fill=(0, 0, 0))
    y_position += 20
    d.text((60, y_position), f"Email: {contact['email']}", font=font, fill=(0, 0, 0))
    y_position += 20
    d.text((60, y_position), f"Phone: {contact['phone']}", font=font, fill=(0, 0, 0))
    y_position += 30
    
    # Requirements
    d.text((40, y_position), "Requirements:", font=font, fill=(0, 0, 0))
    y_position += 20
    req = order_data["details"]["requirements"]
    d.text((60, y_position), f"Employee Count: {req['employee_count']}", font=font, fill=(0, 0, 0))
    y_position += 20
    d.text((60, y_position), f"Frequency: {req['frequency']}", font=font, fill=(0, 0, 0))
    y_position += 20
    d.text((60, y_position), f"Cuisine Preferences: {', '.join(req['cuisine_preferences'])}", font=font, fill=(0, 0, 0))
    y_position += 20
    d.text((60, y_position), f"Special Requests: {req['special_requests']}", font=font, fill=(0, 0, 0))
    y_position += 20
    d.text((60, y_position), f"Arrival Time: {req['arrival_time']}", font=font, fill=(0, 0, 0))
    
    # Save image
    img.save(filepath)
    return filepath