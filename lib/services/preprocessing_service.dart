import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:vector_math/vector_math.dart';

/// Service for preprocessing sensor data before model inference.
/// Loads normalization parameters and applies Z-score normalization.
class PreprocessingService {
  /// Mean values for each feature [Gx, Gy, Gz, Ax, Ay, Az]
  late List<double> mean;
  
  /// Standard deviation for each feature [Gx, Gy, Gz, Ax, Ay, Az]
  late List<double> std;
  
  /// Target gravity vector from training data (normalized)
  /// This is derived from accelerometer means [Ax, Ay, Az]
  late Vector3 targetGravity;
  
  /// Window size in samples (200 = 2 seconds at 100 Hz)
  int windowSize = 200;
  
  /// Sampling rate in Hz
  int samplingRate = 100;
  
  /// Whether parameters have been loaded
  bool _isInitialized = false;
  
  bool get isInitialized => _isInitialized;

  /// Load preprocessing parameters from assets
  Future<void> loadParams() async {
    try {
      final jsonString = await rootBundle.loadString(
        'assets/models/preprocessing_params.json'
      );
      final params = json.decode(jsonString);
      
      mean = List<double>.from(params['mean']);
      std = List<double>.from(params['std']);
      
      // Optional parameters with defaults
      windowSize = params['windows_size'] ?? params['window_size'] ?? 200;
      samplingRate = params['sampling_rate'] ?? 100;
      
      // Auto-detect target gravity vector from accelerometer means
      // Training data mean values tell us where gravity was in that coordinate frame
      final accelMean = Vector3(mean[3], mean[4], mean[5]);
      final gravityMagnitude = accelMean.length;
      
      if (gravityMagnitude > 0.5) {
        // Normalize to unit vector
        targetGravity = accelMean.normalized();
      } else {
        // Fallback: assume gravity on -Y (standard orientation)
        targetGravity = Vector3(0, -1, 0);
        print('Warning: Could not detect gravity axis from means, using default -Y');
      }
      
      _isInitialized = true;
      
      print('PreprocessingService initialized:');
      print('  Mean: $mean');
      print('  Std: $std');
      print('  Window size: $windowSize');
      print('  Sampling rate: $samplingRate Hz');
      print('  Target gravity vector: $targetGravity (magnitude: ${gravityMagnitude.toStringAsFixed(3)})');
    } catch (e) {
      print('Error loading preprocessing params: $e');
      rethrow;
    }
  }

  /// Apply Z-score normalization to raw sensor data
  /// Input: List of samples, each sample is [Gx, Gy, Gz, Ax, Ay, Az]
  /// Output: Normalized samples with same shape
  List<List<double>> normalize(List<List<double>> rawData) {
    if (!_isInitialized) {
      throw StateError('PreprocessingService not initialized. Call loadParams() first.');
    }
    
    return rawData.map((sample) {
      return List.generate(6, (i) {
        // Z-score normalization: (value - mean) / std
        return (sample[i] - mean[i]) / std[i];
      });
    }).toList();
  }

  /// Prepare data for TFLite inference
  /// Converts window to shape [1, 200, 6] (batch, timesteps, features)
  List<List<List<double>>> prepareForInference(List<List<double>> window) {
    if (window.length != windowSize) {
      print('Warning: Window size ${window.length} != expected $windowSize');
    }
    
    final normalized = normalize(window);
    return [normalized]; // Add batch dimension
  }

  /// Validate a single sample has correct format
  bool validateSample(List<double> sample) {
    return sample.length == 6;
  }

  /// Get statistics about a window (for debugging)
  Map<String, dynamic> getWindowStats(List<List<double>> window) {
    if (window.isEmpty) {
      return {'empty': true};
    }
    
    List<double> mins = List.filled(6, double.infinity);
    List<double> maxs = List.filled(6, double.negativeInfinity);
    List<double> sums = List.filled(6, 0.0);
    
    for (var sample in window) {
      for (int i = 0; i < 6; i++) {
        mins[i] = mins[i] < sample[i] ? mins[i] : sample[i];
        maxs[i] = maxs[i] > sample[i] ? maxs[i] : sample[i];
        sums[i] += sample[i];
      }
    }
    
    List<double> means = sums.map((s) => s / window.length).toList();
    
    return {
      'count': window.length,
      'mins': mins,
      'maxs': maxs,
      'means': means,
    };
  }
}
