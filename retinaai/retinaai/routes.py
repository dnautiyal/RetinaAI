
from .prediction import get_prediction, create_output_image
from flask import Flask, request, render_template, flash, request,redirect, url_for
from flask import current_app as app

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Get category of prediction
            category_data = get_prediction(file)
            # Plot the category
            create_output_image(file)
            # Render the result template
            return render_template('result.html', category=category_data, file_name = file.filename)
        return redirect(request.url)
