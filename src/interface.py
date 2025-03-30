import os
import pickle
from flask import Flask, render_template, send_from_directory
from pdf2image import convert_from_path
import argparse

app = Flask(__name__, template_folder='templates', static_folder='static')
from .config import CONVERTED_PDFS_DIR


@app.route('/')
def home():
    previews = []
    for root, _, files in os.walk(CONVERTED_PDFS_DIR):
        if root == '.':
            continue
        pdf_file = next((f for f in files if f.endswith('.pdf')), None)
        if pdf_file:
            try:
                # Convert first page of PDF to image
                preview_path = os.path.join(root, 'preview.jpg')
                if not os.path.exists(preview_path):
                    print("generating preview for", pdf_file)
                    pages = convert_from_path(os.path.join(
                        root, pdf_file), first_page=1, last_page=1)
                    if pages:
                        # Save preview image
                        pages[0].save(preview_path, 'JPEG')
                filename = root.split('\\')[-1]
                previews.append((root, preview_path))
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
    # Sort previews by creation date of the root directory
    previews.sort(key=lambda x: os.path.getctime(x[0]), reverse=True)
    return render_template('home.html', previews=previews)


@app.route('/pdf/<path:folder>')
def view_pdf(folder):
    pdf_file = next((f for f in os.listdir(folder)
                    if f.endswith('.pdf')), None)
    latex_content = []
    callbacks = None
    found = False
    for file in os.listdir(folder):
        if file.endswith('.pkl'):
            with open(os.path.join(folder, file), 'rb') as f:
                latex_content, callbacks = pickle.load(f)
                found = True
                break
    if not found:
        return render_template('error.html', folder=folder)
    pages = []
    if pdf_file:
        num_pages = len([f for f in os.listdir(folder)
                        if f.startswith('page_') and f.endswith('.jpg')])
        if num_pages == 0:
            images = convert_from_path(os.path.join(folder, pdf_file), dpi=300)
            for i, img in enumerate(images):
                page_name = f"page_{i+1}.jpg"
                img_path = os.path.join(folder, page_name)
                img.save(img_path, "JPEG")
                pages.append(page_name)
        else:
            pages = sorted([f for f in os.listdir(folder) if f.startswith(
                'page_') and f.endswith('.jpg')], key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    cost = sum(cb["total_cost"] for cb in callbacks)
    return render_template('pdf_viewer.html', 
                          folder=folder, 
                          pdf_file=pdf_file, 
                          pages=pages, 
                          latex_content=latex_content, 
                          cost=cost)


@app.route('/pdf/<path:folder>/<path:filename>')
def download_pdf(folder, filename):
    return send_from_directory(os.path.abspath(folder), filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask app.')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode')
    args = parser.parse_args()

    app.run(debug=args.debug)
