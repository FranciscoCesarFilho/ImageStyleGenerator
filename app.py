from flask import Flask, render_template, request, url_for
import os
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from model import Generator

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result_image=None)

def process_image(uploaded_file):
    checkpoint = './weights/face_paint_512_v2.pt'
    input_dir = './samples/inputs'
    output_dir = 'static'
    device = 'cpu'
    upsample_align = False
    x32 = False

    # Processamento de imagem
    net = Generator()
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.to(device).eval()

    os.makedirs(output_dir, exist_ok=True)

    filename = uploaded_file.filename
    image_path = os.path.join(input_dir, filename)
    uploaded_file.save(image_path)

    image = load_image(image_path, x32)

    with torch.no_grad():
        image = to_tensor(image).unsqueeze(0) * 2 - 1
        out = net(image.to(device), upsample_align).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)

    result_image_path = os.path.join(output_dir, filename)
    out.save(result_image_path)

    return result_image_path

@app.route('/process', methods=['POST'])
def process():
    result_image = None
    for uploaded_file in request.files.getlist('file'):
        result_image = process_image(uploaded_file)

    if result_image:
        result_image = os.path.relpath(result_image, 'static')

    return render_template('index.html', result_image=result_image)

def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img

if __name__ == '__main__':
    app.run(debug=True)