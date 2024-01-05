import os
#from pyhtml2pdf import converter
from html import escape
import aspose.threed as a3d


def ImportReportExtra():
    filename = "data.txt"  # Replace with the actual file name

    data = {}  # Dictionary to store the extracted values
    scene = a3d.Scene.from_file("Clean body.ply")
    scene.save("output_3d_object_for_stl.obj")

    scene = a3d.Scene.from_file("output_3d_object_for_stl.obj")
    scene.save("CleanStl.stl")
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

    distance_file_path = "Frontdistances.txt"
    with open(distance_file_path, 'r') as distance_file:
        distance_text = distance_file.read()


    # Read text from Backdistances.txt
    distance_file_path_back = "Backdistances.txt"
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
        <img src="kf-new-2022.png" style="margin-top: 80px; width: 250px;height: auto;"/>
       
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
          
            <img src="screenCleanFront.jpg" style="width:1000px;height:auto;"/>               
          
        </tr>
        <tr style="border: none;">
            <img src="screenCleanBack.jpg" style="width:1000px;height:auto;"/>               
        </tr>

    </body>
</html>
    """

    # Save the HTML content to a file
    with open('results.html', 'w') as html_file:
        html_file.write(html_content)
    try:
        path = os.path.abspath('results.html')
        converter.convert(f'file:///{path}', 'sample.pdf')
        print("Conversion successful")
    except Exception as e:
        print(f"Conversion failed: {str(e)}")


ImportReportExtra()
