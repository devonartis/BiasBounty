# Adversarial Testing Documentation

## Overview

This implementation provides comprehensive adversarial testing capabilities to evaluate model robustness. The testing suite includes multiple attack strategies and detailed analysis tools.

## Attack Strategies

### 1. Centroid Attack
```python
def _centroid_attack(self, image, epsilon=0.15):
    """Move image towards opposite cluster centroid"""
    image_flat = image.reshape(1, -1)
    current_pred = self.model.predict(image_flat)[0]
    other_cluster = 1 - current_pred
    direction = self.model.centroids[other_cluster].numpy() - image_flat
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return np.clip(image_flat + epsilon * direction, 0, 1).reshape(image.shape)
```
- Targets cluster boundaries
- Epsilon controls perturbation magnitude
- Maintains image constraints

### 2. Noise Attack
```python
def _noise_attack(self, image, epsilon=0.15):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, epsilon, image.shape)
    return np.clip(image + noise, 0, 1)
```
- Random perturbations
- Controllable noise level
- Natural variation simulation

### 3. Boundary Attack
```python
def _boundary_attack(self, image, steps=5):
    """Binary search along decision boundary"""
    image_flat = image.reshape(1, -1)
    current_pred = self.model.predict(image_flat)[0]
    other_centroid = self.model.centroids[1 - current_pred].numpy()
    # Binary search implementation
```
- Finds minimal perturbation
- Adaptive step size
- Guaranteed success

### 4. Rotation Attack
```python
def _rotation_attack(self, image, max_angle=30):
    """Apply rotation transformation"""
    angle = np.random.uniform(-max_angle, max_angle)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
```
- Geometric transformations
- Random angle selection
- Maintains image content

### 5. Patch Attack
```python
def _patch_attack(self, image, patch_size=0.2):
    """Add adversarial patch"""
    h, w = image.shape[:2]
    patch_h, patch_w = int(h * patch_size), int(w * patch_size)
    patch = np.random.uniform(0, 1, (patch_h, patch_w, 3))
    # Patch application implementation
```
- Localized modifications
- Size-controllable patches
- Position randomization

### 6. Color Attack
```python
def _color_attack(self, image, intensity=0.15):
    """Modify color channels"""
    channel = np.random.randint(0, 3)
    perturbed = image.copy()
    perturbed[:, :, channel] += np.random.uniform(-intensity, intensity)
    return np.clip(perturbed, 0, 1)
```
- Channel-specific changes
- Intensity control
- Natural appearance

### 7. Combined Attack
```python
def _combined_attack(self, image):
    """Apply multiple attack strategies"""
    attacks = [self._centroid_attack, self._noise_attack, 
               self._color_attack, self._rotation_attack]
    num_attacks = np.random.randint(2, len(attacks) + 1)
    selected_attacks = np.random.choice(attacks, num_attacks, replace=False)
    # Combined implementation
```
- Multiple strategy combination
- Random selection
- Enhanced effectiveness

## Usage

### Basic Testing
```bash
python 3.adversarial.py
```
- Single attack strategy
- Basic analysis
- Quick results

### Advanced Suite
```bash
python 4.advance_adversary.py
```
- Multiple attack types
- Comprehensive analysis
- Detailed reporting

## Analysis Tools

### 1. Success Rate Analysis
```python
def evaluate_robustness(original_preds, adversarial_preds):
    changes = np.sum(original_preds != adversarial_preds)
    success_rate = changes / len(original_preds) * 100
    return {
        'total_samples': len(original_preds),
        'prediction_changes': changes,
        'success_rate': success_rate
    }
```
- Attack effectiveness
- Statistical analysis
- Success metrics

### 2. Visual Analysis
```python
def visualize_perturbations(original, perturbed):
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(original)
    plt.title('Original')
    # Visualization implementation
```
- Side-by-side comparison
- Difference visualization
- Pattern analysis

### 3. Robustness Metrics
- Average perturbation magnitude
- Success rate by attack type
- Distance analysis

## Output Files

1. **Adversarial Images**
   - Format: .jpg
   - Location: Adversarial_data/
   - Naming: adv_[attack_type]_[original_name]

2. **Analysis Results**
   - Format: CSV
   - Predictions
   - Success metrics

## Performance Considerations

### 1. Memory Management
- Batch processing
- Resource monitoring
- Efficient storage

### 2. Processing Speed
- Parallel generation
- Optimized computations
- Progress tracking

### 3. Storage Efficiency
- Compressed storage
- Selective saving
- Clean-up utilities

## Best Practices

1. **Attack Configuration**
   - Start with small perturbations
   - Increase gradually
   - Monitor effects

2. **Analysis Approach**
   - Test multiple strategies
   - Compare effectiveness
   - Document findings

3. **Resource Management**
   - Monitor memory usage
   - Batch process large datasets
   - Clean up temporary files

## Future Improvements

1. **Additional Attacks**
   - New strategies
   - Enhanced combinations
   - Targeted approaches

2. **Analysis Tools**
   - Advanced metrics
   - Better visualization
   - Automated reporting

3. **Optimization**
   - Faster generation
   - Better memory usage
   - Enhanced parallelization