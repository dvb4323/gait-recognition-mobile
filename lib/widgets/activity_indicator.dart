import 'package:flutter/material.dart';
import '../models/prediction_result.dart';

/// Visual indicator widget for displaying the current activity.
/// Shows an icon and colored label for the predicted activity.
class ActivityIndicator extends StatelessWidget {
  final PredictionResult? prediction;
  final bool isActive;

  const ActivityIndicator({
    super.key,
    this.prediction,
    this.isActive = false,
  });

  /// Get icon for each activity type
  IconData _getActivityIcon(int classIndex) {
    switch (classIndex) {
      case 0: // Flat Walk
        return Icons.directions_walk;
      case 1: // Up Stairs
        return Icons.stairs;
      case 2: // Down Stairs
        return Icons.stairs_outlined;
      case 3: // Up Slope
        return Icons.trending_up;
      case 4: // Down Slope
        return Icons.trending_down;
      default:
        return Icons.help_outline;
    }
  }

  @override
  Widget build(BuildContext context) {
    final Color activityColor = prediction != null 
        ? Color(prediction!.activityColor)
        : Colors.grey;
    
    final String label = prediction?.activityLabel ?? 'Waiting...';
    final IconData icon = prediction != null 
        ? _getActivityIcon(prediction!.predictedClass)
        : Icons.hourglass_empty;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: activityColor.withOpacity(isActive ? 0.2 : 0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: activityColor.withOpacity(isActive ? 0.5 : 0.2),
          width: 2,
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Activity icon
          AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: activityColor.withOpacity(0.3),
              shape: BoxShape.circle,
            ),
            child: Icon(
              icon,
              size: 64,
              color: activityColor,
            ),
          ),
          const SizedBox(height: 16),
          
          // Activity label
          Text(
            label,
            style: TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
              color: activityColor,
            ),
          ),
          
          // Confidence
          if (prediction != null) ...[
            const SizedBox(height: 8),
            Text(
              'Confidence: ${prediction!.confidencePercent}',
              style: TextStyle(
                fontSize: 18,
                color: activityColor.withOpacity(0.8),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
