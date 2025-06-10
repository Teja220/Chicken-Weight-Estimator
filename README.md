🐔 Chicken Weight Prediction from Images (ResNet18)
This project explores how deep learning can be applied to predict the weight of chickens from their images — combining computer vision with regression modeling. Built as part of a technical challenge by Livestockify, the idea was to create an intuitive, scalable solution for poultry weight estimation using only images.

💡 The Goal
The aim was simple but ambitious:
Can a model look at a chicken and predict its weight in kilograms — just from an image?

With no real-world weight labels available, we simulated a realistic scenario using synthetic data and trained a model using transfer learning with ResNet18.

🗂️ Dataset Creation
Since actual weights weren’t available, we:

Loaded all .jpg images from a directory.

Assigned synthetic weights using numpy.linspace(1.0, 3.0) to evenly spread values.

Added Gaussian noise to mimic real-life variations and imperfections.

A custom PyTorch Dataset class (ChickenWeightDataset) handled all loading, transformation, and pairing of images with their corresponding labels.

🧹 Preprocessing
Images resized to 224×224 pixels to suit ResNet18’s input format.

Applied normalization using standard ImageNet stats.

Data split into 80% training and 20% validation.

🧠 Model Architecture
We used ResNet18, a lightweight but powerful convolutional neural network. Key modifications:

Loaded pretrained weights from ImageNet.

Replaced the final classification layer with a single output neuron for regression.

This allowed the model to output a continuous value — representing the predicted weight of a chicken in kg.

⚙️ Training & Optimization
Loss Function: Mean Squared Error (MSE)

Optimizer: Adam with a learning rate of 0.001

Epochs: 15

At each epoch:

Training and validation losses were recorded.

Gradient updates were performed on training data.

Validation loss was calculated using torch.no_grad() for efficiency.

📊 Evaluation & Visualization
We evaluated model performance using both quantitative and visual metrics.

🔢 R² Score
Achieved an R² score of 0.7041, indicating a strong correlation between predicted and actual weights.

📉 Loss Curves
Plotted training and validation losses over epochs to track model convergence and ensure no overfitting.

🐔 Prediction Samples
Displayed actual vs. predicted weights directly on validation images. This gave a clear, intuitive sense of how well the model was performing visually.

📈 Scatter Plot
Plotted predicted weights against actual weights with a red diagonal line representing ideal predictions. Closer clustering along the line = better accuracy.

📉 Residual Histogram
Visualized the distribution of prediction errors (actual - predicted). A centered histogram around zero suggested unbiased predictions.

✅ Summary
Successfully trained a CNN model to predict chicken weight from images using transfer learning.

Achieved a high R² score and low validation loss.

Synthetic labels worked well as a stand-in for real-world data, providing a solid foundation.

Visual and statistical evaluations confirmed the model’s ability to learn meaningful features related to weight.

📌 Key Takeaways
Transfer learning is a powerful technique even for regression tasks.

Simulated data, when crafted thoughtfully, can be a useful stand-in for real-world scenarios.

This model offers a promising step toward practical, camera-based poultry weight estimation in real farm environments.

Built by: Sai Teja
Challenge Host: Livestockify
Year: 2025
