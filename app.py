import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from algorithms.kmeans import kmeans_clustering
from algorithms.meanshift import meanshift_clustering
from algorithms.dbscan import dbscan_clustering
from algorithms.birch import birch_clustering  # Import the BIRCH clustering function
from algorithms.ward import ward_clustering  # Import the Ward's method clustering function



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder for uploaded images
app.config['RESULT_FOLDER'] = 'static/uploads/'  # Folder for processed images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.secret_key = 'supersecretkey'

# Ensure the upload and result folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Redirect to the algorithm selection page with the filename
            return redirect(url_for('algorithm_selection', filename=filename))
    
    return render_template('upload.html')

# Route for selecting the algorithm
@app.route('/algorithm-selection', methods=['GET', 'POST'])
def algorithm_selection():
    filename = request.args.get('filename')  # Get the filename from query parameters
    if not filename:
        flash('No image file found')
        return redirect(url_for('upload_file'))

    if request.method == 'POST':
        selected_algorithm = request.form.get('algorithm')

        # Redirect to the results page with the selected algorithm and filename
        return redirect(url_for('results', filename=filename, algorithm=selected_algorithm))

    return render_template('algorithm_selection.html', filename=filename)


@app.route('/results', methods=['GET', 'POST'])
def results():
    filename = request.args.get('filename')
    algorithm = request.args.get('algorithm')

    if not filename or not algorithm:
        flash('No image or algorithm selected')
        return redirect(url_for('upload_file'))
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Apply the chosen algorithm
    if algorithm == 'kmeans':
        print("Running K-MEANS...")
        result_image_path = kmeans_clustering(image_path, app.config['RESULT_FOLDER'], K=3)
    elif algorithm == 'meanshift':
        print("Running MEANSHIFT...")
        result_image_path = meanshift_clustering(image_path, app.config['RESULT_FOLDER'])
    elif algorithm == 'dbscan':
        print("Running DBSCAN...")
        result_image_path = dbscan_clustering(image_path, app.config['RESULT_FOLDER'])
    elif algorithm == 'birch':
        print("Running BIRCH...")
        result_image_path = birch_clustering(image_path, app.config['RESULT_FOLDER'])
    elif algorithm == 'ward':
        print("Running WARD...")
        result_image_path = ward_clustering(image_path, app.config['RESULT_FOLDER'])
    else:
        flash('Invalid algorithm selected')
        return redirect(url_for('algorithm_selection', filename=filename))

    return render_template('results.html', original_image=filename, result_image=os.path.basename(result_image_path), algorithm=algorithm)


if __name__ == '__main__':
    app.run(debug=True)
