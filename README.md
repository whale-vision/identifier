# Whale Vision Identifier Backend

## About

The **Whale Vision Identifier Backend** is the neural network and data processing engine behind the Whale Vision Application. It provides identification and embedding extraction for sperm whale images via a WebSocket protocol. This backend enables automated, scalable, and accurate photo-identification, supporting both fluke and flank image recognition.

## Features

- **Deep Learning Inference:** Residual Neural Network models trained with contrastive learning, mapping whale images into a 128-dimensional latent space for rapid and reliable re-identification.
- **GPU Acceleration:** Utilises GPU resources when available for efficient neural network inference; automatically falls back to CPU if necessary.
- **Batch Processing:** Supports processing multiple images in a single request.
- **Dataset Management:** Handles addition of new individuals, confirmation of identifications, and export of organised datasets for research.
- **Logging & Monitoring:** Communicates progress and status updates for transparency and debugging.

## Technologies Used

- Python 3.10+
- PyTorch (for neural network inference)
- WebSockets (for real-time communication)

## Installation

1. **Prerequisites**

- Python 3.10 or higher
- CUDA-enabled GPU (optional, for acceleration)

2. **Setup**

```
git clone https://github.com/whale-vision/identifier
cd identifier
pip install -r requirements.txt
```

3. **Adding Models**
Models are not included in this version, to create your own see whale-vision/identifier-training.
The models should then be placed as:
```
src/models/segmentation.pth

src/models/pm_flank.pth
src/models/pm_fluke.pth

src/models/pm_identification_flank_ind.csv
src/models/pm_identification_fluke_ind.csv
```

4. **Running the Server**

```
python src/websocket.py
```

## WebSocket Protocol

The backend communicates with the frontend using a WebSocket protocol with JSON-formatted messages. Below are the key message types and structures:

### Message Structure

#### Extraction Request  
Frontend → Backend: Request image feature extraction
```
{
  "type": "extract",
  "fileNames": [
    "C:/Whales/Dataset/.../IMG_0689.jpg"
  ]
}
```

**Backend Response:**
```
{
  "type": "extract",
  "data": [{
    "path": "C:/Whales/Dataset/.../IMG_0689.jpg",
    "type": "fluke",
    "embedding": [-0.1445, 0.0332, ...]  // 128D vector
  }]
}
```

#### Identification Request  
Frontend → Backend: Request whale identification
```
{
  "type": "identify",
  "data": [{
    "path": "C:/Whales/Dataset/.../IMG_0689.jpg",
    "type": "fluke",
    "embedding": [-0.1445, 0.0332, ...],
    "identity": "PMOD001_DANTE"  // Optional pre-known identity
  }]
}
```

**Backend Response:**
```
{
  "type": "identify",
  "data": [{
    "path": "C:/Whales/Dataset/.../IMG_0689.jpg",
    "identities": [
      ["PMOD001_DANTE", 0.339],  // [identity, confidence_score]
      ["PMOD096_EDDARD", 0.618]
      // ... Sorted by confidence (lower = better match)
    ]
  }]
}
```

### Status and Progress Messages

The backend will also send plain text status updates for progress tracking:
```
starting on cuda:0 with ../Identifier
segmenting: 1/1
extracting: 1/1
complete!
18:02 25.58
```

### Key Features

- **GPU Acceleration:** Messages will include `"cuda:0"` in logs when using GPU.
- **Batch Processing:** Supports arrays of multiple images per message.
- **Confidence Scores:** Lower values indicate better matches (distance metric).
- **Hybrid Identification:** Combine manual `identity` with automated matching.

## Scientific Context

The Mediterranean sperm whale (*Physeter macrocephalus*) is endangered, and effective conservation relies on accurate, non-invasive individual identification. Traditional manual photo-ID is labor-intensive. Whale Vision leverages machine learning to automate this process, supporting both fluke and (novelly) flank images for identification. Two Residual Neural Network models, trained with contrastive learning, represent each whale in a 128-dimensional latent space for rapid re-identification. Evaluation on the Oceanomare Delphis dataset achieved identification accuracies of 81.2% for fluke images and 76.5% for flank images, demonstrating the effectiveness of this approach. The application is adaptable to other cetacean species, offering a scalable, non-invasive solution to support conservation efforts.

## Citation

If you use this backend in your research, please cite:

> S. Fuller, S. Maggi, T. Kypraios, B. Mussi, and M. Pound, ‘Whale Vision: A Tool for Identifying Sperm Whales and Other Cetaceans by Their Flank or Fluke’, Mar. 07, 2025, Social Science Research Network, Rochester, NY: 5169298. doi: 10.2139/ssrn.5169298.

## License

This project is licensed under the MIT License.  
See the [LICENSE](../LICENSE) file for details.
