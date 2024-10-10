import os
import numpy as np
import cv2
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from algorithms.kmeans import kmeans_clustering
from algorithms.meanshift import meanshift_clustering
from algorithms.dbscan import dbscan_clustering
from algorithms.birch import birch_clustering  # Import the BIRCH clustering function
from algorithms.ward import ward_clustering  # Import the Ward's method clustering function
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from flask import send_from_directory



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder for uploaded images
app.config['RESULT_FOLDER'] = 'static/uploads/'  # Folder for processed images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.secret_key = 'supersecretkey'

for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to serve files from the uploads folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)   

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
            
            return redirect(url_for('algorithm_selection', filename=filename))
    
    return render_template('upload.html')

@app.route('/algorithm-selection', methods=['GET', 'POST'])
def algorithm_selection():
    filename = request.args.get('filename') 
    if not filename:
        flash('No image file found')
        return redirect(url_for('upload_file'))

    if request.method == 'POST':
        selected_algorithm = request.form.get('algorithm')
        use_pca = 'use_pca' in request.form 

        return redirect(url_for('results', filename=filename, algorithm=selected_algorithm, use_pca=use_pca))

    return render_template('algorithm_selection.html', filename=filename)



@app.route('/results')
def results():
    filename = request.args.get('filename')
    algorithm = request.args.get('algorithm')
    use_pca = request.args.get('use_pca') == 'True'  # Convert the string to boolean

    if not filename or not algorithm:
        flash('No image or algorithm selected')
        return redirect(url_for('upload_file'))

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        flash('Image could not be loaded')
        return redirect(url_for('upload_file'))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape  # Store the original shape for reshaping later

    # Prepare the feature vector
    pixel_values = image.reshape((-1, 3)) / 255.0   # Normalize pixel values

    # Get spatial coordinates
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    coordinates = np.stack((yy, xx), axis=2).reshape(-1, 2)
    coordinates = coordinates / np.max(coordinates)  # Normalize coordinates

    # Combine color and spatial information into a feature vector
    feature_vector = np.concatenate((pixel_values, coordinates), axis=1)

    scaler = StandardScaler()
    feature_vector_scaled = scaler.fit_transform(feature_vector)

    # Apply PCA if selected
    if use_pca:
        print("Applying PCA...")
        pca = PCA(n_components=3)
        feature_vector_transformed = pca.fit_transform(feature_vector_scaled)
    else:
        feature_vector_transformed = feature_vector_scaled
        pca = None

    # Choose and apply the clustering algorithm
    result_folder = app.config['RESULT_FOLDER']

    if algorithm == 'kmeans':
        result_image_path = kmeans_clustering(
            feature_vector_transformed, original_shape, scaler, result_folder, use_pca=use_pca, K=5
        )
    elif algorithm == 'meanshift':
        result_image_path = meanshift_clustering(
        feature_vector_transformed, original_shape, scaler, result_folder, use_pca=use_pca, pca=pca
    )
    elif algorithm == 'dbscan':
        result_image_path = dbscan_clustering(
        feature_vector_transformed, original_shape, scaler, result_folder, use_pca=use_pca, pca=pca
    )

    elif algorithm == 'birch':
        result_image_path = birch_clustering(
            feature_vector_transformed, original_shape, scaler, result_folder, use_pca=use_pca, pca=pca, n_clusters=5
        )
    elif algorithm == 'ward':
        result_image_path = ward_clustering(
            image, scaler, result_folder, use_pca=use_pca, pca=pca, n_clusters=20
        )
    else:
        flash('Invalid algorithm selected')
        return redirect(url_for('algorithm_selection', filename=filename))

    return render_template(
        'results.html',
        original_image=filename,
        result_image=os.path.basename(result_image_path),
        algorithm=algorithm
    )



if __name__ == '__main__':
    app.run(debug=True)
