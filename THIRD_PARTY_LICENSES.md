# Third-Party Licenses

This project incorporates several third-party libraries and pre-trained models. Below is a summary of each component, its license, and usage restrictions.

## Pre-trained Models

### OpenAI CLIP (ViT-L/14)
- **Source**: https://github.com/openai/CLIP
- **License**: MIT License
- **Usage**: Image and text embeddings for semantic search
- **Commercial Use**: Allowed

### InsightFace
- **Source**: https://github.com/deepinsight/insightface
- **License**: MIT License
- **Usage**: Face detection and recognition embeddings
- **Commercial Use**: **Non-commercial only for pre-trained models**
- **Note**: The InsightFace library is MIT licensed, but the pre-trained models are for non-commercial research purposes only. Commercial use requires a separate license from InsightFace.

### Qwen2-VL-2B-Instruct
- **Source**: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- **License**: Apache License 2.0
- **Usage**: Vision-language model for image captioning
- **Commercial Use**: Allowed

### 3DDFA_V2
- **Source**: https://github.com/cleardusk/3DDFA_V2
- **License**: MIT License
- **Usage**: 3D face pose estimation (yaw, pitch, roll)
- **Commercial Use**: Allowed
- **Citation**:
  ```
  @inproceedings{guo2020towards,
    title={Towards Fast, Accurate and Stable 3D Dense Face Alignment},
    author={Guo, Jianzhu and Zhu, Xiangyu and Yang, Yang and Yang, Fan and Lei, Zhen and Li, Stan Z},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2020}
  }
  ```

### LAION Aesthetic Predictor
- **Source**: https://github.com/christophschuhmann/improved-aesthetic-predictor
- **License**: MIT License
- **Usage**: Predicting aesthetic quality scores for images
- **Commercial Use**: Allowed
- **Model File**: `sac_logos_ava1-l14-linearMSE.pth`

### Multilingual E5 Large
- **Source**: https://huggingface.co/intfloat/multilingual-e5-large
- **License**: MIT License
- **Usage**: Text embeddings for caption similarity
- **Commercial Use**: Allowed

## Libraries

### XGBoost
- **Source**: https://github.com/dmlc/xgboost
- **License**: Apache License 2.0
- **Usage**: Learning-to-rank model training

### PyTorch
- **Source**: https://pytorch.org/
- **License**: BSD-3-Clause License
- **Usage**: Deep learning framework

### ChromaDB
- **Source**: https://github.com/chroma-core/chroma
- **License**: Apache License 2.0
- **Usage**: Vector database for semantic search

### Hugging Face Transformers
- **Source**: https://github.com/huggingface/transformers
- **License**: Apache License 2.0
- **Usage**: Model loading and inference

### Streamlit
- **Source**: https://github.com/streamlit/streamlit
- **License**: Apache License 2.0
- **Usage**: Web application framework

### OpenCV
- **Source**: https://github.com/opencv/opencv
- **License**: Apache License 2.0
- **Usage**: Image processing

## API Services

### OpenAI API
- **Source**: https://openai.com/
- **Usage**: GPT for natural language query parsing
- **Terms**: Subject to OpenAI Terms of Service
- **Note**: Requires API key; usage incurs costs

---

## Disclaimer

This project is intended for **educational and personal portfolio purposes**. Due to InsightFace's licensing restrictions on pre-trained models, this project should not be used for commercial purposes without obtaining appropriate licenses.

If you plan to use this project commercially, please:
1. Contact InsightFace for commercial licensing
2. Review all third-party licenses for compliance
3. Ensure you have appropriate API usage agreements

## Model Weights

Pre-trained model weights are **not included** in this repository and must be downloaded separately. See the README for setup instructions.
