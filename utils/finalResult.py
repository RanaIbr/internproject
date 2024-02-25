import os
from pyhtml2pdf import converter
from html import escape
import aspose.threed as a3d
# from weasyprint import HTML
from reportlab.pdfgen import canvas


def ImportReportExtra():
    filename = "outputs/text_files/data.txt"  # Replace with the actual file name

    data = {}  # Dictionary to store the extracted values
    scene = a3d.Scene.from_file("resources/3d_models/cleanBody.ply")
    scene.save("outputs/3d_models/output_3d_object_for_stl.obj")

    scene = a3d.Scene.from_file("outputs/3d_models/output_3d_object_for_stl.obj")
    scene.save("outputs/3d_models/CleanStl.stl")
    # Open the file in read mode
    with open(filename, "r") as file:
        # Read each line in the file
        for line in file:
            # Split the line into key and value using the ":" delimiter
            key, value = line.strip().split(":")
            # Remove leading/trailing whitespaces from the key and value
            key = key.strip()
            value = value.strip()
            # Store the key-value pair in the data dictionary
            data[key] = value

    # Extract the values to variables
    name = data.get("name")
    gender = data.get("gender")
    age = int(data.get("age"))
    tall = int(data.get("tall"))
    weight = int(data.get("weight"))
    dimension = data.get("dimension")
    # Read text from Frontdistances.txt

    distance_file_path = "outputs/text_files/FrontdistancesHtml.txt"
    with open(distance_file_path, 'r') as distance_file:
        distance_text = distance_file.read()


    # Read text from Backdistances.txt
    distance_file_path_back = "outputs/text_files/BackdistancesHtml.txt"
    with open(distance_file_path_back, 'r') as distance_file_back:
        distance_text_back = distance_file_back.read()

    css_code = """
        <style>
            .center-image {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            table {
            border-collapse: collapse;
            margin: auto;
        }

        table, th, td {
            border: 1px solid black;
        }
        </style>

    """

    # HTML content with inserted text
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>

        {css_code}
    </head>
    <body>

    <div class="center-image">
        <img src="resources/images/logo_19.jpg" style="margin-top: 80px; width: 250px;height: auto;"/>
       
    </div>
    <div class="center-image" style="margin-top: 45px;">
        <h1>Final Report</h1>
    </div>
    <table style="border: none; margin-top: 10px;">
        <tr style="border: none;">
          
            <td style="border: none; padding: 10px;">

               {distance_text}
               
            </td>
            <td style="border: none;padding: 10px;">
                <!-- Second Cell - Another Table with Example Data -->
  
                 {distance_text_back}

            </td>
        </tr>
      
      <br><br><br><br><br><br><br><br>
    </table>   
        <tr style="border: none;">
          
            <img src="outputs/images/image-front.png" style="width:1000px;height:auto;"/>               
          
        </tr>
        <tr style="border: none;">
            <img src="outputs/images/annotated_image.jpg" style="width:1000px;height:auto;"/>               
        </tr>

        <tr style="border: none;">
          
            <img src="outputs/images/screenCleanFront.jpg" style="width:1000px;height:auto;"/>               
          
        </tr>

    </body>
</html>
    """

    # Save the HTML content to a file
    with open('finalReport.html', 'w') as html_file:
        html_file.write(html_content)
    try:
        path = os.path.abspath('finalReport.html')
        converter.convert(f'file:///{path}', 'finalReport.pdf')

        # HTML(path).write_pdf('sample.pdf')
        print("Conversion successful")
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
