# Logbook

### 3DGS Quantization

Training:
1. Rasterise Gaussians in an image using differentiable Gaussian rasterization.
2. Calculate the loss as a function of the difference between the raster and real terrain images.
3. Adjust the Gaussian parameters according to the loss.
4. Apply automated densification.

Reference: https://www.plainconcepts.com/3d-gaussian-splatting/

Quantized areas:
- Quantized the rendered image + ground truth image
-> Computed loss using these quantized images
- Quantized gradients before and after optimizer steps 

Notes:
- The loss is backpropagated using `loss.backward()` and the gradients of the loss with respect to all parameters that require gradients is computed.
    - Starts with scalar loss value
    - Traverses backwards through all operations used to calculate loss
    - Calculates the gradient for each parameter using chain rule
    - Accumulates these gradients in the `.grad` attribute of each parameter tensor
- The optimiser step is where the model parameters are actually updated based on the computed gradients.
    - The optimiser looks at each parameter in the model.
    - For each parameter, it retrieves the gradient that was computed during backpropagation (stored in `param.grad`)
    - Applies an update rule to adjust the parameter value, adjusting the model parameters in a direction that reduces loss.
    - Clears gradients since PyTorch accumulates gradients, and we want to start fresh for the next iteration.

### Notes

- `xx-nq.py` is the original non-quantised version of the files. For example, `train-nq.py`.
- Attempting to allow quantize transform pass for `gaussian_model.py`.

### Adding forward pass to Gaussian Model

To quantise `gaussian_model.py` using Mase, the model has to be converted into a MaseGraph. This requires the model to be symbolically traceable, which means it needs a forward pass. 