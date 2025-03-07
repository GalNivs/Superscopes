from flask import Flask, render_template, request, jsonify
from superscopes_utils import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/interpret', methods=['POST'])
def interpret_prompt():
    """
    Returns two sets of data in the JSON response:
    1) 'onlyBest': The smaller / best interpretation set
    2) 'showAll': A larger set with more rows or fields per layer

    Both sets contain amplification values for each interpretation.
    Column order (in the front-end) is:
      - Layer
      - Residual Pre MLP
      - MLP Output
      - Hidden State
    """
    data = request.json
    prompt = data.get('prompt', '')
    layer_start = data.get('layerStart', 0)
    layer_end = data.get('layerEnd', 39)
    patch_target = data.get('patchTarget', '0')  # '0' or 'Same Layer'

    return superscopes_analyze(prompt, layer_start, layer_end, patch_target)

if __name__ == '__main__':
    app.run(debug=False)