# ASL_Recognition

Here's the updated `README.md` with a mention that this project was created for a cumulative project for the *Deep Learning with PyTorch* course at Fanshawe College:

---

# Sign Language Recognition Dashboard

This project demonstrates the use of deep learning models to recognize American Sign Language (ASL) letters. It was created as a **cumulative project for the Deep Learning with PyTorch course at Fanshawe College**, showcasing the application of advanced neural network architectures in solving real-world problems.

---

## Features

- **Model Comparison**: Explore ResNet18, ResNet50, and a custom convolutional neural network (CNN) model.
- **Real-Time Predictions**: Get predictions for ASL letters with confidence scores displayed as a bar chart.
- **Custom Image Upload**: Test the models with your own ASL letter images.
- **Dataset Visualization**: View samples from the Sign Language MNIST dataset.
- **Interactive Dashboard**: A user-friendly Gradio interface for seamless interaction.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- Required libraries: `torch`, `torchvision`, `gradio`, `pandas`, `numpy`, `plotly`, `Pillow`

Install the dependencies using pip:

```bash
pip install torch torchvision gradio pandas numpy plotly Pillow
```

---

### Clone the Repository

```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

---

### Running the Application

1. **Download the Dataset**

   - Download the [Sign Language MNIST dataset](https://www.kaggle.com/datamunge/sign-language-mnist).
   - Extract the dataset and place the CSV files (`sign_mnist_train.csv`, `sign_mnist_test.csv`) in the `Extracted_SignLanguageMNIST` folder.

2. **Train the Models**

   If you haven't already trained the models, use the training script provided in the repository to train and save the models (`ResNet18`, `ResNet50`, and the custom CNN).

3. **Launch the Dashboard**

   Run the main script to start the Gradio app:

   ```bash
   python main.py
   ```

4. **Access the Dashboard**

   Open the Gradio app in your browser (usually at `http://127.0.0.1:7860/`).

---

## File Structure

```
sign-language-recognition/
│
├── Extracted_SignLanguageMNIST/
│   ├── sign_mnist_train.csv
│   ├── sign_mnist_test.csv
│
├── saved_models/
│   ├── trained_resnet18.pth
│   ├── trained_resnet50.pth
│   ├── trained_custom.pth
│
├── main.py          # Main script to launch the dashboard
├── README.md        # Project documentation
├── requirements.txt # List of dependencies
```

---

## How It Works

1. **Model Inference**: The selected model processes the input image to predict the ASL letter.
2. **Confidence Visualization**: Confidence scores for all classes are displayed as a bar chart.
3. **Real-Time Updates**: The dashboard updates predictions as you interact with it, providing an intuitive user experience.

---

## Dataset

The models are trained on the [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist) dataset. This dataset contains 28x28 grayscale images of ASL letters, excluding `J` and `Z`, as they involve motion.

---

## Models

### ResNet18

- A deep residual network with 18 layers.
- Designed to handle vanishing gradients effectively using residual connections.

### ResNet50

- A deeper version of ResNet with 50 layers.
- Suitable for large-scale image recognition tasks.

### Custom CNN

- A lightweight convolutional neural network tailored for the ASL recognition task.
- Includes convolutional layers, pooling layers, and fully connected layers.

---

## Purpose

This project was created as a **cumulative project for the Deep Learning with PyTorch course at Fanshawe College**. It demonstrates the application of deep learning techniques in accessibility-focused technology, providing a foundation for further research and development in sign language recognition.

---

## Future Enhancements

- Add support for real-time camera input to recognize ASL letters.
- Implement heatmap visualizations (e.g., Grad-CAM) to interpret model predictions.
- Train models on larger datasets for improved accuracy.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For more information or to contribute to this project, please reach out:

- **Author**: Your Name
- **GitHub**: [yourusername](https://github.com/paigeberrigan)
- **Email**: [paige@interweavemediagroup.ca](mailto:paige@interweavemediagroup.ca)
