# TB-Detection API Microservice



A microservice for automatic tuberculosis detection from chest X-rays using deep learning, with integrated medical context generation powered by LLM technology.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [API Endpoints](#api-endpoints)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [Integration Guide](#integration-guide)
- [Performance](#performance)
- [License](#license)

## Overview

This microservice is a component of a hospital management SaaS platform designed to assist medical professionals in detecting tuberculosis from chest X-ray images. Using a trained deep learning model, it provides rapid classification results along with AI-generated medical context that helps physicians interpret the findings.

## Features

- **Automated TB Detection**: Rapid classification of chest X-rays as normal or showing tuberculosis signs
- **Confidence Scoring**: Probability assessment for each prediction
- **Medical Context Generation**: AI-powered detailed explanations of radiological findings
- **Follow-up Queries**: Interactive Q&A capabilities for specific medical questions related to the analysis
- **REST API Interface**: Simple integration with hospital information systems and medical applications

## API Endpoints

### Image Analysis
- `POST /predict/`: Analyzes uploaded chest X-ray images and classifies them
  - Input: Image file (JPEG, PNG)
  - Output: Classification result and confidence score

### Medical Context
- `POST /ollama_query/`: Generates detailed medical context for detection results
  - Input: Classification results (class and confidence)
  - Output: Structured radiological interpretation with relevant medical context

### Follow-up Queries
- `POST /follow_up/`: Answers follow-up questions related to the analysis
  - Input: User question and previous context
  - Output: Contextually relevant medical response

## Technologies

- **FastAPI**: High-performance web framework for building APIs
- **TensorFlow/Keras**: Deep learning framework for image classification
- **Ollama**: Local large language model for medical context generation
- **PIL/Pillow**: Python Imaging Library for image preprocessing
- **NumPy**: Numerical processing for model inputs and outputs

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/tb-detection-api.git
   cd tb-detection-api
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have Ollama installed and the LLM model available:
   - Install Ollama from [https://ollama.ai/](https://ollama.ai/)
   - Pull the required model: `ollama pull llama3.2`

5. Download the TB detection model:
   - Place the `modelTB.h5` file in the root directory of the project
   
6. Start the API server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Usage

### Analyzing a Chest X-ray

```python
import requests

# API endpoint
url = "http://localhost:8000/predict/"

# Load image file
files = {"file": open("chest_xray.jpg", "rb")}

# Send request
response = requests.post(url, files=files)

# Process results
result = response.json()
print(f"Class: {result['class']}, Confidence: {result['confidence']}")

# Get medical context
context_url = "http://localhost:8000/ollama_query/"
context_response = requests.post(context_url, json=result)
medical_context = context_response.json()
```

### API Response Example

```json
{
  "class": "Tuberculosis",
  "confidence": 0.96421
}
```

### Medical Context Example

```json
{
  "response": "1. Interprétation radiologique: La radiographie montre une opacité apicale droite mal définie associée à des infiltrats nodulaires diffus de taille variable prédominant dans les lobes supérieurs. On note également une perte de volume dans le lobe supérieur droit et des signes possibles de cavitation...[remaining content]"
}
```

## Model Information

The TB detection model (`modelTB.h5`) is a convolutional neural network trained on a dataset of chest X-rays. Key specifications:

- **Architecture**: Custom CNN architecture based on transfer learning
- **Input Size**: 150×150 RGB images (3 channels)
- **Classes**: Binary classification (Normal, Tuberculosis)
- **Training Dataset**: [Appropriate citation of dataset]
- **Validation Accuracy**: ~94% (may vary based on test set)

## Integration Guide

This microservice is designed to integrate with hospital management systems via its REST API. It can be:

1. **Embedded in Radiology Workflows**: Automatically process newly acquired chest X-rays
2. **Integrated with PACS Systems**: Connect to Picture Archiving and Communication Systems
3. **Used in Telemedicine Applications**: Enable remote TB screening capabilities

For secure integration in production:
- Implement proper authentication (JWT, OAuth2)
- Use HTTPS for all communications
- Consider deploying as a containerized service (Docker)

## Performance

- **Inference Time**: ~200-500ms per image (depends on hardware)
- **Throughput**: Capable of processing ~100 images per minute on standard hardware
- **Resource Requirements**:
  - Minimum: 2GB RAM, 2 CPU cores
  - Recommended: 4GB RAM, 4 CPU cores, GPU support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Created by ITNKOC © 2025

*This tool is intended to assist medical professionals and should not be used as the sole basis for diagnostic decisions.*
