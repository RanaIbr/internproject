import pdfkit

# Set the path to the wkhtmltopdf executable
pdfkit_config = pdfkit.configuration(wkhtmltopdf='/path/to/wkhtmltopdf')

# HTML content stored in a variable
html_content = """
<html>
<head><title>Sample HTML</title></head>
<body>
    <h1>Hello, this is HTML content stored in a variable!</h1>
    <p>You can convert this HTML string directly to a PDF using pdfkit.</p>
</body>
</html>
"""

# Path where you want to save the output PDF file
output_pdf = 'output.pdf'

# Configuration options for wkhtmltopdf (optional)
options = {
    'page-size': 'A4',
    'margin-top': '20mm',
    'margin-bottom': '20mm',
}

try:
    # Convert HTML string to PDF using configured path
    pdfkit.from_string(html_content, output_pdf, options=options, configuration=pdfkit_config)
    print(f"PDF generated successfully: {output_pdf}")
except Exception as e:
    print(f"Error generating PDF: {e}")
