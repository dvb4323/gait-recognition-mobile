/// Data model for model prediction results.
class PredictionResult {
  /// Predicted class index (0-4)
  final int predictedClass;
  
  /// Human-readable activity label
  final String activityLabel;
  
  /// Confidence score (0.0 - 1.0)
  final double confidence;
  
  /// Full probability distribution for all classes
  final List<double> probabilities;
  
  /// Timestamp when prediction was made
  final DateTime timestamp;

  /// Activity class labels
  static const Map<int, String> activityLabels = {
    0: 'Flat Walk',
    1: 'Up Stairs',
    2: 'Down Stairs',
    3: 'Up Slope',
    4: 'Down Slope',
  };

  /// Activity colors for UI
  static const Map<int, int> activityColors = {
    0: 0xFF4CAF50, // Green - Flat Walk
    1: 0xFFFF9800, // Orange - Up Stairs
    2: 0xFF2196F3, // Blue - Down Stairs
    3: 0xFF9C27B0, // Purple - Up Slope
    4: 0xFF009688, // Teal - Down Slope
  };

  PredictionResult({
    required this.predictedClass,
    required this.activityLabel,
    required this.confidence,
    required this.probabilities,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  /// Create from model output probabilities
  factory PredictionResult.fromProbabilities(List<double> probs) {
    // Find the class with highest probability
    int maxIndex = 0;
    double maxProb = probs[0];
    
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIndex = i;
      }
    }
    
    return PredictionResult(
      predictedClass: maxIndex,
      activityLabel: activityLabels[maxIndex] ?? 'Unknown',
      confidence: maxProb,
      probabilities: List.from(probs),
    );
  }

  /// Get confidence as percentage string
  String get confidencePercent => '${(confidence * 100).toStringAsFixed(1)}%';

  /// Get color for the predicted activity
  int get activityColor => activityColors[predictedClass] ?? 0xFF9E9E9E;

  @override
  String toString() {
    return 'PredictionResult($activityLabel, confidence: $confidencePercent)';
  }
}
