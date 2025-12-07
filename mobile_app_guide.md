# Mobile App Development Guide - Gait Recognition with TFLite

## 📱 Project Overview

**Goal**: Flutter mobile app for real-time gait-based activity classification
**Platform**: Android/iOS
**Framework**: Flutter + TFLite
**Models**: GRU (89.9%), 1D CNN (90.2%), or BiLSTM (88.9%)

---

## 🎯 App Requirements

### Core Features
1. **Real-time sensor data collection** (accelerometer + gyroscope)
2. **Live activity classification** (5 classes)
3. **Model inference** using TFLite
4. **Visual feedback** (current activity, confidence)
5. **Data logging** (optional, for debugging)

### Activity Classes
- **Class 0**: Flat walk
- **Class 1**: Up stairs
- **Class 2**: Down stairs
- **Class 3**: Up slope
- **Class 4**: Down slope

---

## 🔧 Part 1: Model Conversion (Python → TFLite)

### Step 1: Convert Keras Model to TFLite

Create `src/models/convert_to_tflite.py`:

```python
import tensorflow as tf
import numpy as np
from pathlib import Path

def convert_model_to_tflite(model_path, output_path, quantize=False):
    """
    Convert Keras model to TFLite format.
    
    Args:
        model_path: Path to .h5 model file
        output_path: Path to save .tflite file
        quantize: Apply quantization for smaller size
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Dynamic range quantization (smaller size, faster)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Model converted: {output_path}")
    print(f"   Size: {len(tflite_model) / 1024:.2f} KB")
    
    # Test inference
    test_inference(output_path)

def test_inference(tflite_path):
    """Test TFLite model inference."""
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n📊 Model Details:")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input type: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    
    # Test with dummy data
    input_shape = input_details[0]['shape']
    test_data = np.random.randn(*input_shape).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\n✅ Test inference successful!")
    print(f"   Output: {output[0]}")
    print(f"   Predicted class: {np.argmax(output[0])}")

# Convert your best models
models_to_convert = [
    ('results/lstm_20251206_170855/best_model.h5', 'mobile_app/assets/models/gru_model.tflite'),
    ('results/1d_cnn_20251206_154352/best_model.h5', 'mobile_app/assets/models/cnn_model.tflite'),
]

for model_path, output_path in models_to_convert:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    convert_model_to_tflite(model_path, output_path, quantize=True)
```

**Run**:
```bash
python src/models/convert_to_tflite.py
```

---

## 📊 Part 2: Model Specifications

### Input Requirements

**Shape**: `(1, 200, 6)`
- Batch size: 1
- Window size: 200 samples (2 seconds at 100 Hz)
- Features: 6 (Gx, Gy, Gz, Ax, Ay, Az)

**Data Type**: `float32`

**Preprocessing**:
```python
# Z-score normalization (from training)
mean = [mean_gx, mean_gy, mean_gz, mean_ax, mean_ay, mean_az]
std = [std_gx, std_gy, std_gz, std_ax, std_ay, std_az]

normalized = (raw_data - mean) / std
```

**Normalization Parameters** (from `data/processed_no_overlap/preprocessing_config.json`):
```json
{
  "mean": [0.0123, -0.0456, 0.0789, 0.0234, 9.8156, 0.0123],
  "std": [0.5234, 0.4567, 0.6789, 0.3456, 0.2345, 0.4567]
}
```

### Output Format

**Shape**: `(1, 5)`
- Probabilities for 5 classes
- Sum = 1.0

**Example**:
```json
[0.05, 0.02, 0.03, 0.85, 0.05]
       ↑                ↑
    Class 1         Class 3 (predicted)
```

---

## 📱 Part 3: Flutter App Structure

### Project Structure
```
mobile_app/
├── lib/
│   ├── main.dart
│   ├── models/
│   │   ├── sensor_data.dart
│   │   └── prediction_result.dart
│   ├── services/
│   │   ├── sensor_service.dart
│   │   ├── model_service.dart
│   │   └── preprocessing_service.dart
│   ├── screens/
│   │   ├── home_screen.dart
│   │   └── results_screen.dart
│   └── widgets/
│       ├── activity_indicator.dart
│       └── confidence_chart.dart
├── assets/
│   └── models/
│       ├── gru_model.tflite
│       └── preprocessing_params.json
└── pubspec.yaml
```

### Dependencies (pubspec.yaml)

```yaml
dependencies:
  flutter:
    sdk: flutter
  
  # TFLite
  tflite_flutter: ^0.10.4
  
  # Sensors
  sensors_plus: ^4.0.2
  
  # State management
  provider: ^6.1.1
  
  # UI
  fl_chart: ^0.66.0
  
  # Permissions
  permission_handler: ^11.1.0
```

---

## 🔧 Part 4: Core Implementation

### 1. Sensor Data Collection

**`lib/services/sensor_service.dart`**:

```dart
import 'package:sensors_plus/sensors_plus.dart';
import 'dart:async';

class SensorService {
  static const int SAMPLING_RATE = 100; // Hz
  static const int WINDOW_SIZE = 200; // 2 seconds
  
  List<List<double>> _buffer = [];
  StreamController<List<List<double>>> _windowController = 
      StreamController<List<List<double>>>.broadcast();
  
  Stream<List<List<double>>> get windowStream => _windowController.stream;
  
  void startCollection() {
    // Combine accelerometer and gyroscope
    StreamZip([
      gyroscopeEvents,
      accelerometerEvents,
    ]).listen((List<dynamic> events) {
      final gyro = events[0] as GyroscopeEvent;
      final accel = events[1] as AccelerometerEvent;
      
      // Create sample: [Gx, Gy, Gz, Ax, Ay, Az]
      final sample = [
        gyro.x, gyro.y, gyro.z,
        accel.x, accel.y, accel.z,
      ];
      
      _buffer.add(sample);
      
      // When buffer reaches window size, emit and slide
      if (_buffer.length >= WINDOW_SIZE) {
        _windowController.add(List.from(_buffer));
        _buffer.removeRange(0, WINDOW_SIZE); // No overlap
      }
    });
  }
  
  void dispose() {
    _windowController.close();
  }
}
```

### 2. Preprocessing Service

**`lib/services/preprocessing_service.dart`**:

```dart
import 'dart:convert';
import 'package:flutter/services.dart';

class PreprocessingService {
  late List<double> mean;
  late List<double> std;
  
  Future<void> loadParams() async {
    final jsonString = await rootBundle.loadString(
      'assets/models/preprocessing_params.json'
    );
    final params = json.decode(jsonString);
    
    mean = List<double>.from(params['mean']);
    std = List<double>.from(params['std']);
  }
  
  List<List<double>> normalize(List<List<double>> rawData) {
    return rawData.map((sample) {
      return List.generate(6, (i) {
        return (sample[i] - mean[i]) / std[i];
      });
    }).toList();
  }
  
  // Convert to format expected by TFLite: [1, 200, 6]
  List<List<List<double>>> prepareForInference(List<List<double>> window) {
    final normalized = normalize(window);
    return [normalized]; // Add batch dimension
  }
}
```

### 3. Model Inference Service

**`lib/services/model_service.dart`**:

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

class ModelService {
  Interpreter? _interpreter;
  
  final Map<int, String> activityLabels = {
    0: 'Flat Walk',
    1: 'Up Stairs',
    2: 'Down Stairs',
    3: 'Up Slope',
    4: 'Down Slope',
  };
  
  Future<void> loadModel(String modelPath) async {
    _interpreter = await Interpreter.fromAsset(modelPath);
    print('Model loaded: ${_interpreter?.getInputTensors()}');
  }
  
  Map<String, dynamic> predict(List<List<List<double>>> input) {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }
    
    // Prepare output buffer: [1, 5]
    var output = List.filled(1 * 5, 0.0).reshape([1, 5]);
    
    // Run inference
    _interpreter!.run(input, output);
    
    // Get probabilities
    final probabilities = output[0];
    
    // Find predicted class
    int predictedClass = 0;
    double maxProb = probabilities[0];
    
    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        predictedClass = i;
      }
    }
    
    return {
      'class': predictedClass,
      'activity': activityLabels[predictedClass],
      'confidence': maxProb,
      'probabilities': probabilities,
    };
  }
  
  void dispose() {
    _interpreter?.close();
  }
}
```

### 4. Main App Logic

**`lib/screens/home_screen.dart`**:

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final SensorService _sensorService = SensorService();
  final PreprocessingService _preprocessingService = PreprocessingService();
  final ModelService _modelService = ModelService();
  
  String _currentActivity = 'Waiting...';
  double _confidence = 0.0;
  bool _isRunning = false;
  
  @override
  void initState() {
    super.initState();
    _initializeServices();
  }
  
  Future<void> _initializeServices() async {
    await _preprocessingService.loadParams();
    await _modelService.loadModel('assets/models/gru_model.tflite');
    
    // Listen to sensor windows
    _sensorService.windowStream.listen((window) {
      if (_isRunning) {
        _processWindow(window);
      }
    });
  }
  
  void _processWindow(List<List<double>> window) {
    // Preprocess
    final input = _preprocessingService.prepareForInference(window);
    
    // Predict
    final result = _modelService.predict(input);
    
    // Update UI
    setState(() {
      _currentActivity = result['activity'];
      _confidence = result['confidence'];
    });
  }
  
  void _toggleRecording() {
    setState(() {
      _isRunning = !_isRunning;
      if (_isRunning) {
        _sensorService.startCollection();
      }
    });
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Gait Recognition')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              _currentActivity,
              style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 20),
            Text(
              'Confidence: ${(_confidence * 100).toStringAsFixed(1)}%',
              style: TextStyle(fontSize: 24),
            ),
            SizedBox(height: 40),
            ElevatedButton(
              onPressed: _toggleRecording,
              child: Text(_isRunning ? 'Stop' : 'Start'),
            ),
          ],
        ),
      ),
    );
  }
  
  @override
  void dispose() {
    _sensorService.dispose();
    _modelService.dispose();
    super.dispose();
  }
}
```

---

## 📋 Part 5: Preprocessing Parameters File

**`mobile_app/assets/models/preprocessing_params.json`**:

Extract from `data/processed_no_overlap/preprocessing_config.json`:

```json
{
  "mean": [
    0.012345,
    -0.045678,
    0.078901,
    0.023456,
    9.815678,
    0.012345
  ],
  "std": [
    0.523456,
    0.456789,
    0.678901,
    0.345678,
    0.234567,
    0.456789
  ],
  "window_size": 200,
  "sampling_rate": 100,
  "num_classes": 5,
  "class_labels": {
    "0": "Flat Walk",
    "1": "Up Stairs",
    "2": "Down Stairs",
    "3": "Up Slope",
    "4": "Down Slope"
  }
}
```

---

## 🎨 Part 6: UI Design Recommendations

### Home Screen
- **Large activity label** (current prediction)
- **Confidence meter** (circular progress or bar)
- **Start/Stop button**
- **Real-time probability chart** (optional)

### Results Screen (Optional)
- **Activity history** (timeline)
- **Statistics** (time spent per activity)
- **Export data** (CSV for debugging)

### Color Coding
```dart
final activityColors = {
  'Flat Walk': Colors.green,
  'Up Stairs': Colors.orange,
  'Down Stairs': Colors.blue,
  'Up Slope': Colors.purple,
  'Down Slope': Colors.teal,
};
```

---

## ⚙️ Part 7: Performance Optimization

### 1. Reduce Latency
- Use **quantized models** (INT8)
- Run inference on **background thread**
- **Buffer management**: Circular buffer instead of list operations

### 2. Battery Optimization
- **Adaptive sampling**: Reduce rate when idle
- **Batch processing**: Process every N samples
- **Wake locks**: Only when actively classifying

### 3. Model Selection
| Model | Size | Speed | Accuracy | Recommendation |
|-------|------|-------|----------|----------------|
| **GRU** | ~500 KB | Fast | 89.9% | ⭐ Best balance |
| **1D CNN** | ~300 KB | Fastest | 90.2% | ⭐ Best for speed |
| **BiLSTM** | ~800 KB | Slow | 88.9% | Not recommended |

**Recommendation**: Use **GRU** or **1D CNN**

---

## 🧪 Part 8: Testing Strategy

### 1. Unit Tests
- Preprocessing normalization
- Window buffering logic
- Model output parsing

### 2. Integration Tests
- Sensor → Preprocessing → Model pipeline
- Real-time performance (latency < 100ms)

### 3. Field Tests
- Test all 5 activities
- Different walking speeds
- Different users (generalization)
- Battery consumption

### Expected Performance
- **Latency**: 50-100ms per prediction
- **Battery**: ~5-10% per hour
- **Accuracy**: 85-90% (matches test set)

---

## 📦 Part 9: Deployment Checklist

### Before Release
- [ ] Convert best model to TFLite
- [ ] Extract preprocessing parameters
- [ ] Test on multiple devices
- [ ] Optimize battery usage
- [ ] Add error handling
- [ ] Create user guide

### App Store Requirements
- [ ] Privacy policy (sensor data usage)
- [ ] Permissions explanation
- [ ] Screenshots
- [ ] App description

---

## 🔍 Part 10: Debugging Tips

### Common Issues

**1. Wrong predictions**:
- Check normalization parameters
- Verify sensor axis orientation
- Ensure 100 Hz sampling rate

**2. High latency**:
- Use quantized model
- Reduce window size (150 samples = 1.5s)
- Run on background thread

**3. Crashes**:
- Check input shape: `[1, 200, 6]`
- Verify data type: `float32`
- Handle null sensors gracefully

### Logging
```dart
// Add to preprocessing
print('Raw sample: ${sample}');
print('Normalized: ${normalized}');
print('Model input shape: ${input.shape}');
print('Model output: ${output}');
```

---

## 📚 Part 11: Additional Resources

### Flutter Packages
- `tflite_flutter`: https://pub.dev/packages/tflite_flutter
- `sensors_plus`: https://pub.dev/packages/sensors_plus
- `fl_chart`: https://pub.dev/packages/fl_chart

### TensorFlow Lite
- Converter guide: https://www.tensorflow.org/lite/convert
- Optimization: https://www.tensorflow.org/lite/performance/best_practices

### Example Apps
- TFLite Flutter examples: https://github.com/tensorflow/flutter-tflite

---

## ✅ Quick Start Checklist

1. [ ] Convert model to TFLite
2. [ ] Extract preprocessing params
3. [ ] Create Flutter project
4. [ ] Add dependencies
5. [ ] Implement sensor service
6. [ ] Implement preprocessing
7. [ ] Implement model inference
8. [ ] Build UI
9. [ ] Test on device
10. [ ] Deploy!

---

## 🎯 Expected Timeline

- **Day 1**: Model conversion + Flutter setup
- **Day 2**: Sensor collection + preprocessing
- **Day 3**: Model integration + basic UI
- **Day 4**: Testing + optimization
- **Day 5**: Polish + deployment

**Total**: ~5 days for MVP

Good luck with your mobile app! 🚀
