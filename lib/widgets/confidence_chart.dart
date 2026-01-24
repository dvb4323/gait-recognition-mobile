import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../models/prediction_result.dart';

/// Bar chart showing probability distribution across all activity classes.
class ConfidenceChart extends StatelessWidget {
  final PredictionResult? prediction;

  const ConfidenceChart({
    super.key,
    this.prediction,
  });

  @override
  Widget build(BuildContext context) {
    if (prediction == null) {
      return const Center(
        child: Text(
          'No prediction yet',
          style: TextStyle(color: Colors.grey),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: BarChart(
        BarChartData(
          alignment: BarChartAlignment.spaceAround,
          maxY: 1.0,
          minY: 0,
          barTouchData: BarTouchData(
            enabled: true,
            touchTooltipData: BarTouchTooltipData(
              getTooltipItem: (group, groupIndex, rod, rodIndex) {
                final label = PredictionResult.activityLabels[group.x] ?? '';
                final value = (rod.toY * 100).toStringAsFixed(1);
                return BarTooltipItem(
                  '$label\n$value%',
                  const TextStyle(color: Colors.white),
                );
              },
            ),
          ),
          titlesData: FlTitlesData(
            show: true,
            topTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
            rightTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                getTitlesWidget: (value, meta) {
                  final labels = ['Flat', 'Up↑', 'Down↓', 'Slope↑', 'Slope↓'];
                  final index = value.toInt();
                  if (index >= 0 && index < labels.length) {
                    return Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Text(
                        labels[index],
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: index == prediction!.predictedClass
                              ? FontWeight.bold
                              : FontWeight.normal,
                          color: index == prediction!.predictedClass
                              ? Color(prediction!.activityColor)
                              : Colors.grey,
                        ),
                      ),
                    );
                  }
                  return const Text('');
                },
              ),
            ),
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 40,
                getTitlesWidget: (value, meta) {
                  if (value == 0 || value == 0.5 || value == 1.0) {
                    return Text(
                      '${(value * 100).toInt()}%',
                      style: const TextStyle(fontSize: 10, color: Colors.grey),
                    );
                  }
                  return const Text('');
                },
              ),
            ),
          ),
          gridData: FlGridData(
            show: true,
            horizontalInterval: 0.25,
            getDrawingHorizontalLine: (value) {
              return FlLine(
                color: Colors.grey.withOpacity(0.2),
                strokeWidth: 1,
              );
            },
            drawVerticalLine: false,
          ),
          borderData: FlBorderData(show: false),
          barGroups: _buildBarGroups(),
        ),
        swapAnimationDuration: const Duration(milliseconds: 150),
      ),
    );
  }

  List<BarChartGroupData> _buildBarGroups() {
    final probs = prediction?.probabilities ?? List.filled(5, 0.0);
    
    return List.generate(5, (index) {
      final isSelected = index == prediction?.predictedClass;
      final color = Color(PredictionResult.activityColors[index] ?? 0xFF9E9E9E);
      
      return BarChartGroupData(
        x: index,
        barRods: [
          BarChartRodData(
            toY: probs[index],
            color: isSelected ? color : color.withOpacity(0.4),
            width: 28,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(6)),
            backDrawRodData: BackgroundBarChartRodData(
              show: true,
              toY: 1,
              color: Colors.grey.withOpacity(0.1),
            ),
          ),
        ],
      );
    });
  }
}
