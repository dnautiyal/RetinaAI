
from .prediction import get_category, plot_category
from flask import request, render_template
from flask import current_app as app


@app.route('/', methods=['GET', 'POST'])
def retina_ai():
    # Write the GET Method to get the index file
    if request.method == 'GET':
        return render_template('index.html')
    # Write the POST Method to post the results file
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('File Not Uploaded')
            return
        # Read file from upload
        file = request.files['file']
        # Get category of prediction
        category = get_category(img=file)
        # Plot the category
        plot_category(file)
        # Render the result template
        return render_template('result.html', category=category)
