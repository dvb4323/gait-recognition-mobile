import 'package:flutter/services.dart';
import '../models/prediction_result.dart';

/// Native TFLite model service using Platform Channels.
/// Uses native Android TFLite with FlexDelegate for SELECT_TF_OPS support.
/// This enables GRU/LSTM models to work properly.
class NativeModelService {
  static const MethodChannel _channel = MethodChannel(
    'com.example.gait_recognition_flutter/tflite',
  );

  /// Whether model is loaded and ready
  bool _isInitialized = false;
  String? _currentModel;
  List<int>? _inputShape;
  List<int>? _outputShape;

  bool get isInitialized => _isInitialized;
  String? get currentModel => _currentModel;
  List<int>? get inputShape => _inputShape;
  List<int>? get outputShape => _outputShape;

  /// Load TFLite model from assets using native Android TFLite
  /// Supports SELECT_TF_OPS for GRU/LSTM models
  Future<void> loadModel([String modelPath = 'gait_lstm_model.tflite']) async {
    try {
      final result = await _channel.invokeMethod<Map>('loadModel', {
        'modelPath': modelPath,
      });

      if (result != null && result['success'] == true) {
        _isInitialized = true;
        _currentModel = modelPath;
        _inputShape = result['inputShape']?.cast<int>();
        _outputShape = result['outputShape']?.cast<int>();

        print('NativeModelService: Model loaded successfully');
        print('  Model: $modelPath');
        print('  Input shape: $_inputShape');
        print('  Output shape: $_outputShape');
      } else {
        throw Exception('Model loading returned unsuccessful');
      }
    } on PlatformException catch (e) {
      print('NativeModelService: Error loading model: ${e.message}');
      rethrow;
    }
  }

  /// Run inference on preprocessed input
  /// Input shape: [1, 200, 6] (batch, timesteps, features)
  /// Output: PredictionResult with activity label and confidence
  Future<PredictionResult> predict(List<List<List<double>>> input) async {
    if (!_isInitialized) {
      throw StateError('NativeModelService not initialized. Call loadModel() first.');
    }

    final stopwatch = Stopwatch()..start();

    try {
      final result = await _channel.invokeMethod<Map>('predict', {
        'input': input,
      });

      stopwatch.stop();

      if (result != null && result['success'] == true) {
        final probabilities = (result['probabilities'] as List).cast<double>();

        print('NativeModelService: Inference took ${stopwatch.elapsedMilliseconds}ms');
        print('  Raw output: $probabilities');

        return PredictionResult.fromProbabilities(probabilities);
      } else {
        throw Exception('Prediction returned unsuccessful');
      }
    } on PlatformException catch (e) {
      print('NativeModelService: Prediction error: ${e.message}');
      rethrow;
    }
  }

  /// Check if model is loaded (from native side)
  Future<bool> checkModelLoaded() async {
    try {
      final result = await _channel.invokeMethod<bool>('isModelLoaded');
      return result ?? false;
    } catch (e) {
      return false;
    }
  }

  /// Close the model and release resources
  Future<void> dispose() async {
    try {
      await _channel.invokeMethod('closeModel');
      _isInitialized = false;
      _currentModel = null;
      _inputShape = null;
      _outputShape = null;
      print('NativeModelService: Disposed');
    } catch (e) {
      print('NativeModelService: Error disposing: $e');
    }
  }
}
